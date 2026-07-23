from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from repro.aicrowd_whestbench.run_whest_acronym_pipeline import (
    _autoevolve_env,
    _filter_prompt,
    _method_initials,
    _proposer_prompt,
    _validate_filter,
    configured_codes,
    generate_codes,
    validate_proposal,
)
from repro.aicrowd_whestbench.run_whest_acronym_sweep import (
    code_from_index,
    index_from_code,
    next_codes,
)


class WhestAcronymPipelineTest(unittest.TestCase):
    def test_generate_codes_returns_unique_three_letter_codes(self):
        codes = generate_codes(100)

        self.assertEqual(len(codes), 100)
        self.assertEqual(len(set(codes)), 100)
        self.assertTrue(all(len(code) == 3 and code.isalpha() for code in codes))
        self.assertTrue(all(code == code.upper() for code in codes))

    def test_proposer_prompt_uses_assigned_code_and_only_supplied_task(self):
        prompt = _proposer_prompt("QRS", "TASK SENTINEL")

        self.assertIn("assigned", prompt)
        self.assertIn("QRS", prompt)
        self.assertIn("TASK SENTINEL", prompt)
        self.assertIn("exactly three", prompt)
        self.assertNotIn("exact_radial", prompt)

    def test_configured_codes_preserve_caller_traversal_order(self):
        environment = {
            "RUN_WHEST_ACRONYM_CODES": "AAA,AAB,AAC",
            "RUN_WHEST_ACRONYM_CODE_SOURCE": "lexicographic_traversal",
        }
        with patch.dict("os.environ", environment, clear=True):
            codes, source = configured_codes(99)

        self.assertEqual(codes, ["AAA", "AAB", "AAC"])
        self.assertEqual(source, "lexicographic_traversal")

    def test_lexicographic_sweep_codes_and_skips(self):
        self.assertEqual(code_from_index(0), "AAA")
        self.assertEqual(code_from_index(25), "AAZ")
        self.assertEqual(code_from_index(26), "ABA")
        self.assertEqual(index_from_code("ZZZ"), 26**3 - 1)

        codes, cursor = next_codes(0, 4, {"AAB"})
        self.assertEqual(codes, ["AAA", "AAC", "AAD", "AAE"])
        self.assertEqual(code_from_index(cursor), "AAF")

    def test_proposal_requires_exact_acronym_expansion(self):
        proposal = {
            "code": "RAP",
            "method_name": "Radial Antithetic Projection",
            "one_line": "A sufficiently concrete one-line technical summary.",
            "mode_prompt": "x" * 500,
        }

        validate_proposal(proposal, "RAP")
        self.assertEqual(_method_initials(proposal["method_name"]), "RAP")
        with self.assertRaisesRegex(ValueError, "does not expand"):
            validate_proposal({**proposal, "method_name": "Wrong Acronym Name"}, "RAP")

    def test_filter_must_partition_all_valid_codes(self):
        verdict = {
            "selected": [
                {"code": "ABC", "reason": "a" * 50},
                {"code": "DEF", "reason": "b" * 50},
            ],
            "rejected": [{"code": "GHI", "reason": "c" * 50}],
            "global_reasoning": "d" * 100,
        }

        self.assertEqual(
            _validate_filter(verdict, ["ABC", "DEF", "GHI"], 2),
            ["ABC", "DEF"],
        )
        with self.assertRaisesRegex(ValueError, "cover every"):
            _validate_filter(
                {**verdict, "rejected": [{"code": "JKL", "reason": "c" * 50}]},
                ["ABC", "DEF", "GHI"],
                2,
            )

    def test_filter_prompt_contains_all_proposals(self):
        proposals = [
            {
                "code": "ABC",
                "method_name": "Adaptive Block Control",
                "one_line": "one",
                "mode_prompt": "mode one",
            },
            {
                "code": "DEF",
                "method_name": "Directional Even Features",
                "one_line": "two",
                "mode_prompt": "mode two",
            },
        ]

        prompt = _filter_prompt(proposals, "TASK SENTINEL", 1)

        self.assertIn("single independent filter", prompt)
        self.assertIn("Select exactly 1", prompt)
        self.assertIn("ABC", prompt)
        self.assertIn("DEF", prompt)
        self.assertIn("TASK SENTINEL", prompt)

    def test_autoevolve_followup_is_one_full_100_mlp_iteration(self):
        env = _autoevolve_env(
            code="ABC",
            tag="test",
            run_root=Path("/tmp/test-run"),
            mode_file=Path("/tmp/abc.md"),
            starter=Path("/tmp/starter.py"),
        )

        self.assertEqual(env["WHEST_SEARCH_N_MLPS"], "100")
        self.assertEqual(env["RUN_WHEST_AUTO_NUM_EPOCHS"], "1")
        self.assertEqual(env["RUN_WHEST_AUTO_CLI_TIMEOUT"], "21600")


if __name__ == "__main__":
    unittest.main()
