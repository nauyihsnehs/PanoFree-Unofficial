import argparse
import json
import os
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from panofree.phase5 import run_phase5


def main():
    default_config = os.path.join(ROOT, "configs", "phase5.example.json")
    parser = argparse.ArgumentParser(description="Run PanoFree Phase 5 pipeline.")
    parser.add_argument(
        "--config",
        required=False,
        default=default_config,
        help="Path to the Phase 5 JSON config.",
    )
    args = parser.parse_args()

    result = run_phase5(args.config)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
