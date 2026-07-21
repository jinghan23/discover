import tempfile
import unittest
from pathlib import Path

from ttt_discover import State
from ttt_discover.algorithms.abmcts.sampler import ABMCTSSampler


class DummyEnv:
    state_type = State

    @classmethod
    def create_initial_state(cls, problem_type: str) -> State:
        del problem_type
        return State(
            timestep=-1,
            construction=["root"],
            code="root",
            value=-0.25,
            metadata={"official_suite": {"n_mlps": 10}},
        )


class ABMCTSRootWidthTest(unittest.TestCase):
    def _sampler(self, tmp: str, **kwargs) -> ABMCTSSampler:
        return ABMCTSSampler(
            file_path=str(Path(tmp) / "abmcts_sampler.json"),
            env_type=DummyEnv,
            actions=("local_refine", "analytic_moments"),
            seed=0,
            **kwargs,
        )

    def test_forced_root_width_round_robins_actions_within_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            sampler = self._sampler(tmp, root_min_width=4)

            parents = sampler.sample_states(4)

        metadata = [parent.metadata["abmcts"] for parent in parents]
        self.assertEqual([item["node_id"] for item in metadata], [0, 0, 0, 0])
        self.assertEqual(
            [item["action"] for item in metadata],
            ["local_refine", "analytic_moments"] * 2,
        )

    def test_successful_unique_children_satisfy_minimum_width(self):
        with tempfile.TemporaryDirectory() as tmp:
            sampler = self._sampler(tmp, root_min_width=2)
            parents = sampler.sample_states(2)
            children = [
                State(0, ["child-a"], "child-a", -0.2),
                State(0, ["child-b"], "child-b", -0.1),
            ]

            sampler.update_states(children, parents, save=False)
            stats = sampler.get_sample_stats()

        self.assertEqual(stats["abmcts/root_width"], 2)
        self.assertEqual(stats["abmcts/root_width_satisfied"], 1)
        self.assertEqual(children[0].metadata["official_suite"]["n_mlps"], 10)

    def test_reward_scale_and_root_width_survive_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            sampler = self._sampler(
                tmp,
                root_min_width=3,
                reward_scale=10.0,
                prior_mean=-2.5,
            )
            self.assertEqual(sampler._nodes[0].score, -2.5)
            sampler.flush(step=3)

            restored = self._sampler(
                tmp,
                root_min_width=0,
                reward_scale=1.0,
                resume_step=3,
            )

        self.assertEqual(restored.root_min_width, 3)
        self.assertEqual(restored.reward_scale, 10.0)


if __name__ == "__main__":
    unittest.main()
