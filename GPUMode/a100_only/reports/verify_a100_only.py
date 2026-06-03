from __future__ import annotations

import ast
import difflib
import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
A100_ROOT = ROOT / "GPUMode" / "a100_only"
REPRO_ROOT = A100_ROOT / "reproductions"
TOP_ROOT = ROOT / "GPUMode" / "top_solutions"
REPORT_ROOT = A100_ROOT / "reports"


TOP_A100 = {
    "matmul_v2": "top1_A100_submission_780718.py",
    "trimul": "top1_A100_submission_380716.py",
    "vectoradd_v2": "top1_A100_submission_779892.py",
    "vectorsum_v2": "top1_A100_submission_779823.py",
}


def trimul_case():
    torch.manual_seed(7)
    batch, seqlen, dim, hidden = 2, 3, 8, 4
    weights = {
        "norm.weight": torch.randn(dim),
        "norm.bias": torch.randn(dim),
        "left_proj.weight": torch.randn(hidden, dim) / hidden**0.5,
        "right_proj.weight": torch.randn(hidden, dim) / hidden**0.5,
        "left_gate.weight": torch.randn(hidden, dim) / hidden**0.5,
        "right_gate.weight": torch.randn(hidden, dim) / hidden**0.5,
        "out_gate.weight": torch.randn(hidden, dim) / hidden**0.5,
        "to_out_norm.weight": torch.randn(hidden),
        "to_out_norm.bias": torch.randn(hidden),
        "to_out.weight": torch.randn(dim, hidden) / dim**0.5,
    }
    return (
        torch.randn(batch, seqlen, seqlen, dim),
        torch.randint(0, 2, (batch, seqlen, seqlen), dtype=torch.float32),
        weights,
        {"dim": dim, "hidden_dim": hidden},
    )


def trimul_ref(data):
    x, mask, weights, config = data
    dim, hidden = config["dim"], config["hidden_dim"]
    x = torch.nn.functional.layer_norm(x, (dim,), weights["norm.weight"], weights["norm.bias"])
    left = x @ weights["left_proj.weight"].t()
    right = x @ weights["right_proj.weight"].t()
    left = left * torch.sigmoid(x @ weights["left_gate.weight"].t()) * mask.unsqueeze(-1)
    right = right * torch.sigmoid(x @ weights["right_gate.weight"].t()) * mask.unsqueeze(-1)
    out = torch.einsum("bikd,bjkd->bijd", left, right)
    out = torch.nn.functional.layer_norm(out, (hidden,), weights["to_out_norm.weight"], weights["to_out_norm.bias"])
    out = out * torch.sigmoid(x @ weights["out_gate.weight"].t())
    return out @ weights["to_out.weight"].t()


PROBLEMS = {
    "matmul_v2": {
        "cases": [
            lambda: (
                torch.randn(8, 16, dtype=torch.float16),
                torch.randn(16, 12, dtype=torch.float16),
                torch.empty(8, 12, dtype=torch.float16),
            )
        ],
        "ref": lambda data: data[0] @ data[1],
        "atol": 1e-3,
        "rtol": 1e-3,
    },
    "vectoradd_v2": {
        "cases": [
            lambda: (
                torch.randn(127, 127, dtype=torch.float16),
                torch.randn(127, 127, dtype=torch.float16),
                torch.empty(127, 127, dtype=torch.float16),
            ),
            lambda: (
                torch.randn(128, 128, dtype=torch.float16),
                torch.randn(128, 128, dtype=torch.float16),
                torch.empty(128, 128, dtype=torch.float16),
            ),
        ],
        "ref": lambda data: data[0] + data[1],
        "atol": 1e-3,
        "rtol": 1e-3,
    },
    "vectorsum_v2": {
        "cases": [
            lambda: (torch.randn(1025, dtype=torch.float32) * 2 + 3, torch.empty(1, dtype=torch.float32)),
            lambda: (torch.randn(4096, dtype=torch.float32) - 9, torch.empty(1, dtype=torch.float32)),
        ],
        "ref": lambda data: data[0].to(torch.float64).sum().to(torch.float32),
        "atol": 1e-2,
        "rtol": 1e-5,
    },
    "trimul": {
        "cases": [trimul_case],
        "ref": trimul_ref,
        "atol": 2e-2,
        "rtol": 2e-2,
    },
}


def import_submission(problem: str):
    submission = REPRO_ROOT / problem / "submission.py"
    problem_files = TOP_ROOT / problem / "problem_files"
    with tempfile.TemporaryDirectory(prefix=f"a100_verify_{problem}_") as tmp:
        tmp_path = Path(tmp)
        for file in problem_files.iterdir():
            if file.is_file() and file.suffix == ".py":
                name = file.name.replace("shared_", "", 1)
                shutil.copy2(file, tmp_path / name)
        shutil.copy2(submission, tmp_path / "submission.py")
        old_path = list(sys.path)
        sys.path.insert(0, str(tmp_path))
        try:
            spec = importlib.util.spec_from_file_location("submission", tmp_path / "submission.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules.pop("submission", None)
            assert spec and spec.loader
            spec.loader.exec_module(module)
            return module
        finally:
            sys.path[:] = old_path
            for name in ("task", "utils", "reference", "submission"):
                sys.modules.pop(name, None)


def has_custom_kernel(path: Path) -> bool:
    tree = ast.parse(path.read_text())
    return any(isinstance(node, ast.FunctionDef) and node.name == "custom_kernel" for node in tree.body)


def sequence_similarity(problem: str, submission: Path) -> float:
    top = TOP_ROOT / problem / "submissions" / TOP_A100[problem]
    return difflib.SequenceMatcher(None, submission.read_text(), top.read_text(), autojunk=False).ratio()


def run_problem(problem: str) -> dict:
    submission = REPRO_ROOT / problem / "submission.py"
    result = {
        "problem": problem,
        "submission": str(submission.relative_to(ROOT)),
        "a100_top1": str((TOP_ROOT / problem / "submissions" / TOP_A100[problem]).relative_to(ROOT)),
        "cuda_available": torch.cuda.is_available(),
        "checks": {},
        "cpu_smoke": [],
    }
    try:
        compile(submission.read_text(), str(submission), "exec")
        result["checks"]["syntax"] = True
        result["checks"]["custom_kernel_def"] = has_custom_kernel(submission)
        result["checks"]["similarity_to_a100_top1"] = sequence_similarity(problem, submission)
    except Exception as exc:
        result["checks"]["syntax"] = False
        result["checks"]["error"] = repr(exc)
        return result

    try:
        module = import_submission(problem)
        result["checks"]["import"] = True
    except Exception as exc:
        result["checks"]["import"] = False
        result["checks"]["import_error"] = repr(exc)
        return result

    spec = PROBLEMS[problem]
    for idx, make_case in enumerate(spec["cases"]):
        data = make_case()
        case = {"case": idx}
        try:
            cloned = tuple(x.clone() if isinstance(x, torch.Tensor) else x for x in data)
            out = module.custom_kernel(cloned)
            ref = spec["ref"](data)
            case["ran"] = True
            case["allclose"] = bool(torch.allclose(out, ref, atol=spec["atol"], rtol=spec["rtol"]))
            case["max_abs_error"] = float((out.float() - ref.float()).abs().max().item())
        except Exception as exc:
            case["ran"] = False
            case["error"] = repr(exc)
        result["cpu_smoke"].append(case)
    return result


def main() -> None:
    results = [run_problem(problem) for problem in sorted(PROBLEMS)]
    out = REPORT_ROOT / "verification_results.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    print(out)
    for item in results:
        print(json.dumps(item, ensure_ascii=False))


if __name__ == "__main__":
    main()
