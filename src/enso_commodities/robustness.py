from __future__ import annotations

import pandas as pd


def summarize_candidate_robustness(
    specification_results: pd.DataFrame,
    fragility_summary: pd.DataFrame,
    *,
    expected_specifications: int,
) -> pd.DataFrame:
    required = {
        "index_definition",
        "anchor_type",
        "commodity",
        "mean_return",
        "bootstrap_bh_q_value",
        "placebo_bh_q_value",
        "passes_specification",
        "stable_direction",
    }
    if not required.issubset(specification_results.columns):
        raise ValueError(
            f"Robustness results are missing columns: {sorted(required - set(specification_results))}"
        )
    if specification_results.duplicated(["index_definition", "anchor_type", "commodity"]).any():
        raise ValueError("Robustness specification keys must be unique")
    if not {"commodity", "survives_all_deletions"}.issubset(fragility_summary.columns):
        raise ValueError("Fragility summary is missing required columns")

    fragility = fragility_summary.loc[:, ["commodity", "survives_all_deletions"]]
    merged = specification_results.merge(
        fragility, on="commodity", how="left", validate="many_to_one"
    )
    rows: list[dict[str, object]] = []
    for commodity, group in merged.groupby("commodity", sort=True, observed=True):
        weakest = group.loc[
            group[["bootstrap_bh_q_value", "placebo_bh_q_value"]].max(axis=1).idxmax()
        ]
        specifications = len(group)
        passes = int(group["passes_specification"].sum())
        fragility_pass = bool(group["survives_all_deletions"].iloc[0])
        rows.append(
            {
                "commodity": commodity,
                "specifications_tested": specifications,
                "specifications_passed": passes,
                "min_mean_return": float(group["mean_return"].min()),
                "max_mean_return": float(group["mean_return"].max()),
                "stable_direction_all": bool(group["stable_direction"].all()),
                "worst_bootstrap_q_value": float(group["bootstrap_bh_q_value"].max()),
                "worst_placebo_q_value": float(group["placebo_bh_q_value"].max()),
                "weakest_index_definition": weakest["index_definition"],
                "weakest_anchor_type": weakest["anchor_type"],
                "primary_fragility_pass": fragility_pass,
                "passes_timing_index_robustness": (
                    fragility_pass
                    and specifications == expected_specifications
                    and passes == expected_specifications
                ),
            }
        )
    return pd.DataFrame.from_records(rows)


def summarize_control_robustness(
    specification_results: pd.DataFrame,
    *,
    expected_specifications: int,
) -> pd.DataFrame:
    required = {
        "index_definition",
        "anchor_type",
        "commodity",
        "reject_bootstrap_raw",
        "reject_placebo_raw",
        "stable_direction",
    }
    if not required.issubset(specification_results.columns):
        raise ValueError(
            f"Control robustness results are missing columns: {sorted(required - set(specification_results))}"
        )
    rows: list[dict[str, object]] = []
    for commodity, group in specification_results.groupby("commodity", sort=True, observed=True):
        specifications = len(group)
        bootstrap_rejections = int(group["reject_bootstrap_raw"].sum())
        placebo_rejections = int(group["reject_placebo_raw"].sum())
        rows.append(
            {
                "commodity": commodity,
                "specifications_tested": specifications,
                "bootstrap_raw_rejections": bootstrap_rejections,
                "placebo_raw_rejections": placebo_rejections,
                "stable_direction_all": bool(group["stable_direction"].all()),
                "fails_specificity_everywhere": (
                    specifications == expected_specifications
                    and bootstrap_rejections == expected_specifications
                    and placebo_rejections == expected_specifications
                ),
            }
        )
    return pd.DataFrame.from_records(rows)
