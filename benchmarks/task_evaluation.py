"""Aggregate bounded runtime logs without invoking an LLM."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_core.evaluation import evaluate_logs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_logs(args.logs)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"tasks": report["tasks"], "completion_rate": report["completion_rate"]}))


if __name__ == "__main__":
    main()
