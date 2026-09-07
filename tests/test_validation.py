from __future__ import annotations

from enso_commodities.validation import evidence_label, load_validation_contract


def test_v2_contract_is_complete_and_explicitly_post_discovery() -> None:
    contract = load_validation_contract()
    assert contract.version == 2
    assert contract.discovery_snapshot == "2026-09-06"
    assert contract.commodities == ("Coconut oil", "Palm oil", "Rubber, RSS3")
    assert contract.maximum_shift_years == 60


def test_evidence_labels_do_not_confuse_low_power_with_equivalence() -> None:
    assert evidence_label(
        robust_association=False,
        confidence_lower=-0.3,
        confidence_upper=0.4,
        equivalence_bound=0.1,
        below_mde=True,
    ) == "inconclusive_underpowered"
    assert evidence_label(
        robust_association=False,
        confidence_lower=-0.04,
        confidence_upper=0.08,
        equivalence_bound=0.1,
        below_mde=False,
    ) == "evidence_against_material_effect"

