from __future__ import annotations

import numpy as np
import pandas as pd

from enso_commodities.program_timing_null import (
    fit_program_alignment,
    program_null_p_values,
    summarize_program_alignments,
)
from enso_commodities.validation import load_validation_contract


def test_program_alignment_enumerates_the_locked_family() -> None:
    contract = load_validation_contract()
    dates = pd.date_range("1970-01-01", periods=720, freq="MS")
    phase = np.arange(len(dates)) % 36
    roni = np.where(phase < 8, 0.8, -0.1)
    enso = pd.DataFrame({"date": dates, "roni": roni, "oni": roni * 0.9})
    rows = []
    generator = np.random.default_rng(3)
    for commodity in contract.commodities:
        for date in dates:
            rows.append(
                {
                    "date": date,
                    "commodity": commodity,
                    "seasonal_adjusted_log_return": generator.normal(0, 0.02),
                    "market_adjusted_log_return": generator.normal(0, 0.02),
                }
            )
    cells = fit_program_alignment(
        pd.DataFrame(rows), enso, contract, shift_years=0, minimum_valid_episodes=10
    )
    expected = (
        len(contract.commodities)
        * len(contract.timing_indexes)
        * len(contract.timing_anchors)
        * len(contract.timing_horizons)
        * len(contract.timing_outcomes)
    )
    assert len(cells) == expected
    shifted = cells.copy()
    shifted["shift_years"] = 1
    shifted["is_observed_timing"] = False
    shifted["t_statistic"] *= 0.5
    summaries = summarize_program_alignments(pd.concat([cells, shifted], ignore_index=True))
    assert set(program_null_p_values(summaries)) == {
        "median_absolute_t_p_value",
        "maximum_absolute_t_p_value",
        "positive_estimate_share_p_value",
    }

