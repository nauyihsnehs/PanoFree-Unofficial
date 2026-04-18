import argparse
import json
import os
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from panofree.phase3 import run_phase3


def main():
    default_config = os.path.join(ROOT, "configs", "phase3.example.json")
    parser = argparse.ArgumentParser(description="Run PanoFree Phase 3 pipeline.")
    parser.add_argument(
        "--config",
        required=False,
        default=default_config,
        help="Path to the Phase 3 JSON config.",
    )
    args = parser.parse_args()

    result = run_phase3(args.config)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
