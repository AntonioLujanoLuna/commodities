import pandas as pd
import pytest

from enso_commodities.universe import (
    CommodityRegistry,
    InferenceSpec,
    load_commodity_registry,
    validate_registry,
)


def small_registry() -> CommodityRegistry:
    return CommodityRegistry(
        inference=InferenceSpec(
            primary_anchor="retrospective",
            primary_horizon_months=12,
            primary_return="market_adjusted_cumulative_return",
            alternative="two_sided",
            minimum_valid_episodes=10,
            fdr_family="mechanism_candidates_only",
        ),
        entries=pd.DataFrame(
            [
                {
                    "commodity": "Cocoa",
                    "role": "mechanism_candidate",
                    "group": "beverages",
                    "selection_note": "weather pathway",
                },
                {
                    "commodity": "Gold",
                    "role": "negative_control",
                    "group": "precious_metals",
                    "selection_note": "negative control",
                },
                {
                    "commodity": "Copper",
                    "role": "excluded",
                    "group": None,
                    "selection_note": "no pathway",
                },
            ]
        ),
        expected_counts={
            "mechanism_candidates": 1,
            "negative_controls": 1,
            "excluded": 1,
            "source_total": 3,
        },
    )


def test_project_registry_has_frozen_expected_counts() -> None:
    registry = load_commodity_registry()
    counts = registry.entries["role"].value_counts().to_dict()
    assert counts == {
        "excluded": 36,
        "mechanism_candidate": 32,
        "negative_control": 3,
    }
    assert registry.entries["commodity"].nunique() == 71


def test_registry_requires_exact_source_coverage() -> None:
    registry = small_registry()
    validate_registry(
        registry,
        source_commodities={"Cocoa", "Gold", "Copper"},
        allowed_horizons={0, 6, 12},
        allowed_anchors={"retrospective", "observable"},
    )

    with pytest.raises(ValueError, match="does not exactly cover"):
        validate_registry(
            registry,
            source_commodities={"Cocoa", "Gold", "Copper", "Silver"},
            allowed_horizons={0, 6, 12},
            allowed_anchors={"retrospective", "observable"},
        )


def test_registry_rejects_duplicate_assignment() -> None:
    registry = small_registry()
    duplicate = pd.concat([registry.entries, registry.entries.iloc[[0]]], ignore_index=True)
    invalid = CommodityRegistry(
        inference=registry.inference,
        entries=duplicate,
        expected_counts=registry.expected_counts,
    )
    with pytest.raises(ValueError, match="contains duplicates"):
        validate_registry(
            invalid,
            source_commodities={"Cocoa", "Gold", "Copper"},
            allowed_horizons={12},
            allowed_anchors={"retrospective"},
        )
