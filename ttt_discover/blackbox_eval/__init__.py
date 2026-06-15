"""Blackbox verifier service utilities for autonomous Codex workspaces.

This package is intentionally decoupled from any specific environment. The
server owns the trusted verifier; workspace clients only submit candidate code.
"""

from ttt_discover.blackbox_eval.client_template import build_eval_client_source

__all__ = ["build_eval_client_source"]
