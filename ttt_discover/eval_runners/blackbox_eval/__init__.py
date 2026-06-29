"""Blackbox verifier service utilities for autonomous Codex workspaces."""


def build_eval_client_source(*args, **kwargs):
    from ttt_discover.eval_runners.blackbox_eval.client_template import (
        build_eval_client_source as _build_eval_client_source,
    )

    return _build_eval_client_source(*args, **kwargs)

__all__ = ["build_eval_client_source"]
