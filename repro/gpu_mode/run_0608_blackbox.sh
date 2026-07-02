#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_0608_TARGET=autoevolve_blackbox exec bash "$SCRIPT_DIR/run_0608.sh" "$@"
