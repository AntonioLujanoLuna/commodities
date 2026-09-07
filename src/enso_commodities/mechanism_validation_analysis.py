"""Receipt-writing v2 mechanism stage for externally supplied validation inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .config import project_root
from .mechanism_validation import MechanismInputs, fit_mechanism_chain
from .provenance import sha256_file, write_json_atomic


def run_mechanism_validation(
    weather_path: Path,
    yield_path: Path,
    supply_revision_path: Path,
    *,
    output_dir: Path,
    config_path: Path | None = None,
) -> Path:
    mechanism_path = config_path or project_root() / "config" / "mechanism_v2.yaml"
    with mechanism_path.open(encoding="utf-8") as handle:
        config: dict[str, Any] = yaml.safe_load(handle)
    inputs = MechanismInputs(
        weather=pd.read_csv(weather_path, parse_dates=["date"]),
        yields=pd.read_csv(yield_path),
        supply_revisions=pd.read_csv(
            supply_revision_path, parse_dates=["information_date", "price_date"]
        ),
    )
    expected = {str(value) for value in config["commodities"]}
    observed = set(inputs.weather["commodity"]) & set(inputs.yields["commodity"]) & set(
        inputs.supply_revisions["commodity"]
    )
    if observed != expected:
        raise ValueError(
            f"Mechanism validation commodity set differs from the frozen contract: "
            f"expected {sorted(expected)}, observed {sorted(observed)}"
        )
    results = fit_mechanism_chain(
        inputs, minimum_clusters=int(config["requirements"]["minimum_clusters_per_link"])
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "mechanism_v2_results.csv"
    results.to_csv(results_path, index=False)
    link_passes = {
        f"{commodity}:{link}": bool(group["passes_link"].iloc[0])
        for (commodity, link), group in results.groupby(["commodity", "link"], observed=True)
    }
    receipt = {
        "data_provenance": "external_validation_inputs",
        "scope": str(config["scope"]),
        "complete_chain_by_commodity": {
            commodity: bool(group["passes_link"].all())
            for commodity, group in results.groupby("commodity", observed=True)
        },
        "link_passes": link_passes,
        "input_hashes": {
            weather_path.name: sha256_file(weather_path),
            yield_path.name: sha256_file(yield_path),
            supply_revision_path.name: sha256_file(supply_revision_path),
            mechanism_path.name: sha256_file(mechanism_path),
        },
        "output_hashes": {results_path.name: sha256_file(results_path)},
    }
    write_json_atomic(output_dir / "mechanism_v2_summary.json", receipt)
    return output_dir

