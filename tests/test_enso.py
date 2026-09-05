import pandas as pd

from enso_commodities.enso import construct_cold_episodes, construct_warm_episodes


def test_construct_warm_episodes_requires_persistence_and_tracks_signal_date() -> None:
    dates = pd.date_range("1999-12-01", periods=14, freq="MS")
    values = [0.1, 0.5, 0.7, 0.8, 0.6, 0.9, 0.2, 0.5, 0.6, 0.7, 0.8, 0.4, 0.1, 0.0]
    data = pd.DataFrame({"date": dates, "roni": values})

    episodes = construct_warm_episodes(
        data,
        index_name="roni",
        threshold=0.5,
        minimum_duration_months=5,
        observable_delay_after_center_months=2,
    )

    assert len(episodes) == 1
    episode = episodes.iloc[0]
    assert episode["onset_date"] == pd.Timestamp("2000-01-01")
    assert episode["persistence_date"] == pd.Timestamp("2000-05-01")
    assert episode["observable_date"] == pd.Timestamp("2000-07-01")
    assert episode["end_date"] == pd.Timestamp("2000-05-01")
    assert episode["duration_months"] == 5
    assert episode["peak_date"] == pd.Timestamp("2000-05-01")
    assert episode["peak_value"] == 0.9


def test_construct_warm_episodes_breaks_on_missing_calendar_month() -> None:
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2000-01-01", "2000-02-01", "2000-04-01", "2000-05-01", "2000-06-01"]
            ),
            "roni": [0.6] * 5,
        }
    )

    episodes = construct_warm_episodes(
        data,
        index_name="roni",
        threshold=0.5,
        minimum_duration_months=5,
        observable_delay_after_center_months=2,
    )

    assert episodes.empty


def test_construct_cold_episodes_uses_negative_threshold_and_minimum_peak() -> None:
    data = pd.DataFrame(
        {
            "date": pd.date_range("2000-01-01", periods=6, freq="MS"),
            "roni": [-0.5, -0.7, -1.0, -0.8, -0.6, 0.0],
        }
    )
    episodes = construct_cold_episodes(
        data,
        index_name="roni",
        threshold=-0.5,
        minimum_duration_months=5,
        observable_delay_after_center_months=2,
    )
    assert len(episodes) == 1
    episode = episodes.iloc[0]
    assert episode["episode_id"] == "roni_cold_2000_01"
    assert episode["direction"] == "cold"
    assert episode["peak_value"] == -1.0
