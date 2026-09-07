from __future__ import annotations

import argparse
from pathlib import Path

from enso_commodities.mechanism_validation_analysis import run_mechanism_validation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the v2 regional mechanism validation.")
    parser.add_argument("--weather", type=Path, required=True)
    parser.add_argument("--yields", type=Path, required=True)
    parser.add_argument("--supply-revisions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    print(
        run_mechanism_validation(
            args.weather,
            args.yields,
            args.supply_revisions,
            output_dir=args.output,
            config_path=args.config,
        )
    )
