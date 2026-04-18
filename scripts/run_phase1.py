import argparse
import json
import os
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from panofree.phase1 import run_phase1


def main():
    parser = argparse.ArgumentParser(description="Run PanoFree Phase 1 pipeline.")
    parser.add_argument(
        "--config",
        required=False, default='../configs/phase1.example.json',
        help="Path to the Phase 1 JSON config.",
    )
    args = parser.parse_args()

    result = run_phase1(args.config)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

