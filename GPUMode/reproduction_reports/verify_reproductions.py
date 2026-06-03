from __future__ import annotations

import ast
import difflib
import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path
from types import ModuleType

import torch


ROOT = Path(__file__).resolve().parents[2]
REPRO_ROOT = ROOT / "GPUMode" / "reproductions"
TOP_ROOT = ROOT / "GPUMode" / "top_solutions"
REPORT_ROOT = ROOT / "GPUMode" / "reproduction_reports"


PROBLEMS = {
    "trimul": {
        "cpu_cases": [],
        "reference": None,
        "atol": 2e-2,
        "rtol": 2e-2,
    },
    "matmul_v2": {
        "cpu_cases": [
            lambda: (
                torch.randn(8, 16, dtype=torch.float16),
                torch.randn(16, 12, dtype=torch.float16),
                torch.empty(8, 12, dtype=torch.float16),
            )
        ],
        "reference": lambda data: data[0] @ data[1],
        "atol": 1e-3,
        "rtol": 1e-3,
    },
    "vectoradd_v2": {
        "cpu_cases": [
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
        "reference": lambda data: data[0] + data[1],
        "atol": 1e-3,
        "rtol": 1e-3,
    },
    "vectorsum_v2": {
        "cpu_cases": [
            lambda: (
                torch.randn(1023, dtype=torch.float32) * 3.0 + 4.0,
                torch.empty(1, dtype=torch.float32),
            ),
            lambda: (
                torch.randn(4096, dtype=torch.float32) * 0.2 - 17.0,
                torch.empty(1, dtype=torch.float32),
            ),
        ],
        "reference": lambda data: data[0].to(torch.float64).sum().to(torch.float32),
        "atol": 1e-2,
        "rtol": 1e-5,
    },
}


def make_trimul_case() -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], dict[str, int]]:
    torch.manual_seed(123)
    batch, seqlen, dim, hidden_dim = 1, 4, 8, 4
    weights = {
        "norm.weight": torch.randn(dim, dtype=torch.float32),
        "norm.bias": torch.randn(dim, dtype=torch.float32),
        "left_proj.weight": torch.randn(hidden_dim, dim, dtype=torch.float32) / hidden_dim**0.5,
        "right_proj.weight": torch.randn(hidden_dim, dim, dtype=torch.float32) / hidden_dim**0.5,
        "left_gate.weight": torch.randn(hidden_dim, dim, dtype=torch.float32) / hidden_dim**0.5,
        "right_gate.weight": torch.randn(hidden_dim, dim, dtype=torch.float32) / hidden_dim**0.5,
        "out_gate.weight": torch.randn(hidden_dim, dim, dtype=torch.float32) / hidden_dim**0.5,
        "to_out_norm.weight": torch.randn(hidden_dim, dtype=torch.float32),
        "to_out_norm.bias": torch.randn(hidden_dim, dtype=torch.float32),
        "to_out.weight": torch.randn(dim, hidden_dim, dtype=torch.float32) / dim**0.5,
    }
    return (
        torch.randn(batch, seqlen, seqlen, dim, dtype=torch.float32),
        torch.randint(0, 2, (batch, seqlen, seqlen), dtype=torch.float32),
        weights,
        {"dim": dim, "hidden_dim": hidden_dim},
    )


def trimul_reference(data):
    input_tensor, mask, weights, config = data
    dim, hidden_dim = config["dim"], config["hidden_dim"]
    x = torch.nn.functional.layer_norm(
        input_tensor,
        (dim,),
        weights["norm.weight"],
        weights["norm.bias"],
        eps=1e-5,
    )
    left = x @ weights["left_proj.weight"].t()
    right = x @ weights["right_proj.weight"].t()
    left_gate = torch.sigmoid(x @ weights["left_gate.weight"].t())
    right_gate = torch.sigmoid(x @ weights["right_gate.weight"].t())
    out_gate = torch.sigmoid(x @ weights["out_gate.weight"].t())
    mask_e = mask.unsqueeze(-1)
    left = left * left_gate * mask_e
    right = right * right_gate * mask_e
    out = torch.einsum("bikd,bjkd->bijd", left, right)
    out = torch.nn.functional.layer_norm(
        out,
        (hidden_dim,),
        weights["to_out_norm.weight"],
        weights["to_out_norm.bias"],
        eps=1e-5,
    )
    out = out * out_gate
    return out @ weights["to_out.weight"].t()


PROBLEMS["trimul"]["cpu_cases"] = [make_trimul_case]
PROBLEMS["trimul"]["reference"] = trimul_reference


def import_submission(problem: str, submission: Path) -> ModuleType:
    problem_files = TOP_ROOT / problem / "problem_files"
    with tempfile.TemporaryDirectory(prefix=f"verify_{problem}_") as tmp:
        tmp_path = Path(tmp)
        for file in problem_files.iterdir():
            if file.is_file() and file.suffix == ".py":
                name = file.name
                if name.startswith("shared_"):
                    name = name.replace("shared_", "", 1)
                shutil.copy2(file, tmp_path / name)
        shutil.copy2(submission, tmp_path / "submission.py")
        old_path = list(sys.path)
        sys.path.insert(0, str(tmp_path))
        try:
            spec = importlib.util.spec_from_file_location("submission", tmp_path / "submission.py")
            if spec is None or spec.loader is None:
                raise RuntimeError("could not build import spec")
            module = importlib.util.module_from_spec(spec)
            sys.modules.pop("submission", None)
            spec.loader.exec_module(module)
            return module
        finally:
            sys.path[:] = old_path
            sys.modules.pop("task", None)
            sys.modules.pop("utils", None)
            sys.modules.pop("reference", None)
            sys.modules.pop("submission", None)


def has_custom_kernel(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return any(isinstance(node, ast.FunctionDef) and node.name == "custom_kernel" for node in tree.body)


def similarity(problem: str, submission: Path) -> dict[str, float | str]:
    repro_text = submission.read_text(encoding="utf-8")
    best_ratio = -1.0
    best_file = ""
    for top in (TOP_ROOT / problem / "submissions").glob("*.py"):
        ratio = difflib.SequenceMatcher(
            None,
            repro_text,
            top.read_text(encoding="utf-8"),
            autojunk=False,
        ).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_file = str(top.relative_to(ROOT))
    return {"closest_top1_file": best_file, "sequence_similarity": best_ratio}


def verify_problem(problem: str) -> dict:
    submission = REPRO_ROOT / problem / "submission.py"
    notes = REPRO_ROOT / problem / "NOTES.md"
    result = {
        "problem": problem,
        "submission": str(submission.relative_to(ROOT)) if submission.exists() else None,
        "notes": str(notes.relative_to(ROOT)) if notes.exists() else None,
        "cuda_available": torch.cuda.is_available(),
        "checks": {},
        "cpu_smoke": [],
        "similarity": None,
    }

    if not submission.exists():
        result["checks"]["exists"] = False
        return result

    result["checks"]["exists"] = True
    try:
        compile(submission.read_text(encoding="utf-8"), str(submission), "exec")
        result["checks"]["syntax"] = True
    except Exception as exc:
        result["checks"]["syntax"] = False
        result["checks"]["syntax_error"] = repr(exc)
        return result

    try:
        result["checks"]["custom_kernel_def"] = has_custom_kernel(submission)
    except Exception as exc:
        result["checks"]["custom_kernel_def"] = False
        result["checks"]["custom_kernel_error"] = repr(exc)

    result["similarity"] = similarity(problem, submission)

    try:
        module = import_submission(problem, submission)
        result["checks"]["import"] = True
        result["checks"]["custom_kernel_callable"] = callable(getattr(module, "custom_kernel", None))
    except Exception as exc:
        result["checks"]["import"] = False
        result["checks"]["import_error"] = repr(exc)
        return result

    spec = PROBLEMS.get(problem)
    if spec is None:
        result["checks"]["cpu_smoke_supported"] = False
        return result

    for idx, make_case in enumerate(spec["cpu_cases"]):
        case_result = {"case": idx}
        data = make_case()
        try:
            out = module.custom_kernel(tuple(x.clone() if isinstance(x, torch.Tensor) else x for x in data))
            ref = spec["reference"](data)
            case_result["ran"] = True
            case_result["shape"] = list(out.shape) if isinstance(out, torch.Tensor) else None
            case_result["dtype"] = str(out.dtype) if isinstance(out, torch.Tensor) else str(type(out))
            case_result["allclose"] = bool(torch.allclose(out, ref, atol=spec["atol"], rtol=spec["rtol"]))
            if isinstance(out, torch.Tensor):
                case_result["max_abs_error"] = float((out.float() - ref.float()).abs().max().item())
        except Exception as exc:
            case_result["ran"] = False
            case_result["error"] = repr(exc)
        result["cpu_smoke"].append(case_result)
    return result


def main() -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    problems = sorted(p.name for p in REPRO_ROOT.iterdir() if p.is_dir()) if REPRO_ROOT.exists() else []
    results = [verify_problem(problem) for problem in problems]
    out = REPORT_ROOT / "verification_results.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(out)
    for result in results:
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
