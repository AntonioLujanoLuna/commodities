from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from enso_commodities.disruption_sources import (
    audit_disruption_source_readiness,
    load_disruption_source_audit,
)


def test_live_audit_blocks_the_chain_for_specific_source_contract_failures() -> None:
    audit = audit_disruption_source_readiness()

    assert audit["status"] == "blocked_source_contract"
    assert not audit["complete_chain_ready"]
    assert audit["links_ready"] == {
        "cold_phase_to_regional_rainfall": False,
        "rainfall_to_throughput": False,
        "throughput_shortfall_to_price": False,
    }
    assert (
        audit["source_diagnostics"]["transport_nsw_port_of_newcastle"]["calendar_year_clusters"]
        == 9
    )
    assert audit["source_diagnostics"]["nqbp_hay_point_terminals"]["calendar_year_clusters"] == 7
    assert "world_bank_pink_sheet" in audit["eligible_sources"]["price"]


def test_a_source_is_not_vintage_dated_merely_because_it_has_enough_years(
    tmp_path: Path,
) -> None:
    raw = deepcopy(load_disruption_source_audit())
    newcastle = raw["sources"]["transport_nsw_port_of_newcastle"]
    newcastle["first_observation"] = "2010-01"
    path = tmp_path / "sources.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    audit = audit_disruption_source_readiness(path)

    assert audit["eligible_sources"]["throughput"] == ["transport_nsw_port_of_newcastle"]
    assert audit["eligible_sources"]["vintage_throughput"] == []
    assert not audit["links_ready"]["throughput_shortfall_to_price"]


def test_audit_rejects_a_non_calendar_cluster_unit(tmp_path: Path) -> None:
    raw = deepcopy(load_disruption_source_audit())
    raw["cluster_unit"] = "quarter"
    path = tmp_path / "sources.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="calendar-year"):
        load_disruption_source_audit(path)
