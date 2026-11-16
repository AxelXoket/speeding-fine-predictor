"""Synthetic radar records labelled with the 2026 speeding rules.

No public per-driver fine records exist, so the records are generated: vehicle, road and measured speed are
sampled, and the labels come from rules.penalty_for. A small, documented share of rows is then damaged the way
real entry data gets damaged, so the cleaning step in the notebook has something real to fix.
"""
import pathlib

import numpy as np
import pandas as pd

from .rules import penalty_for, speed_limit, tiers_for

COLUMNS = (
    "kayit_no",
    "arac_tipi",
    "yol_tipi",
    "hiz_siniri",
    "olculen_hiz",
    "saat",
    "hava",
    "ceza_tutari_tl",
    "ehliyet_geri_alma_gun",
)
WEATHER_TYPES = ("acik", "yagmurlu", "sisli", "karli")
WEATHER_LABELS = {"acik": "Açık", "yagmurlu": "Yağmurlu", "sisli": "Sisli", "karli": "Karlı"}

# Sampling shares are assumptions chosen to give every fine tier enough rows, not traffic statistics.
VEHICLE_SHARE = {
    "otomobil": 0.5,
    "panelvan": 0.06,
    "kamyonet": 0.11,
    "motosiklet": 0.08,
    "minibus": 0.05,
    "otobus": 0.07,
    "kamyon": 0.13,
}
ROAD_SHARE = {"yerlesim_yeri": 0.45, "sehirlerarasi_cift_yonlu": 0.2, "bolunmus_yol": 0.2, "otoyol": 0.15}
WEATHER_SHARE = (0.7, 0.18, 0.07, 0.05)
# Bands: under the limit, over the limit but below the first fine, then one weight per fine tier.
UNDER_LIMIT_SHARE = 0.34
TOLERANCE_SHARE = 0.12
TIER_SHARES = (0.12, 0.09, 0.08, 0.06, 0.05, 0.04, 0.04, 0.03, 0.03)
UNDER_LIMIT_MIN_EXCESS = -25
OPEN_TIER_WIDTH = 25

MISSING_SPEED_ROWS = 6
IMPOSSIBLE_SPEED_ROWS = 3
MISSING_WEATHER_ROWS = 15
MESSY_TEXT_ROWS = 12
DUPLICATE_ROWS = 8
IMPOSSIBLE_SPEEDS = (-48, 0, 412)

DEFAULT_ROWS = 600
DEFAULT_SEED = 2026
DEFAULT_CSV = pathlib.Path(__file__).resolve().parent.parent / "data" / "speeding_records.csv"


def _excess_bands(road: str) -> list[tuple[int, int]]:
    """Inclusive excess ranges in band order: under limit, tolerance, then each fine tier."""
    tiers = tiers_for(road)
    bands = [(UNDER_LIMIT_MIN_EXCESS, 0), (1, tiers[0].min_excess - 1)]
    for tier in tiers:
        upper = tier.max_excess if tier.max_excess is not None else tier.min_excess + OPEN_TIER_WIDTH - 1
        bands.append((tier.min_excess, upper))
    return bands


def _band_weights(road: str) -> np.ndarray:
    weights = np.array((UNDER_LIMIT_SHARE, TOLERANCE_SHARE, *TIER_SHARES), dtype=float)
    if len(weights) != len(_excess_bands(road)):
        raise ValueError("TIER_SHARES must have one weight per fine tier")
    return weights / weights.sum()


def _check_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an int")


def generate_records(n_rows: int = DEFAULT_ROWS, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """Return n_rows clean, labelled records; the same seed always gives the same table."""
    _check_seed(seed)
    if n_rows < 1:
        raise ValueError("n_rows must be at least 1")
    rng = np.random.default_rng(seed)
    vehicles = rng.choice(list(VEHICLE_SHARE), size=n_rows, p=list(VEHICLE_SHARE.values()))
    roads = rng.choice(list(ROAD_SHARE), size=n_rows, p=list(ROAD_SHARE.values()))
    rows = []
    for number, (vehicle, road) in enumerate(zip(vehicles, roads), start=1):
        limit = speed_limit(vehicle, road)
        bands = _excess_bands(road)
        low, high = bands[rng.choice(len(bands), p=_band_weights(road))]
        speed = limit + int(rng.integers(low, high + 1))
        penalty = penalty_for(vehicle, road, speed)
        rows.append(
            (
                number,
                str(vehicle),
                str(road),
                limit,
                speed,
                int(rng.integers(0, 24)),
                str(rng.choice(WEATHER_TYPES, p=WEATHER_SHARE)),
                penalty.fine_tl,
                penalty.suspension_days,
            )
        )
    frame = pd.DataFrame(rows, columns=list(COLUMNS))
    frame["olculen_hiz"] = frame["olculen_hiz"].astype("Int64")
    return frame


def _messy(text: str, variant: int) -> str:
    return (f" {text.capitalize()}", f"{text.upper()} ", f"{text.title()}  ")[variant % 3]


def add_dirty_rows(clean: pd.DataFrame, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """Return a copy of clean with missing values, impossible speeds, messy text and duplicate rows."""
    _check_seed(seed)
    damaged_count = MISSING_SPEED_ROWS + IMPOSSIBLE_SPEED_ROWS + MISSING_WEATHER_ROWS + MESSY_TEXT_ROWS
    if len(clean) < damaged_count + DUPLICATE_ROWS:
        raise ValueError("table is too small to damage")
    rng = np.random.default_rng(seed + 1)
    dirty = clean.reset_index(drop=True).copy()  # row positions below are used as labels
    picked = rng.permutation(len(dirty))
    cuts = np.cumsum((MISSING_SPEED_ROWS, IMPOSSIBLE_SPEED_ROWS, MISSING_WEATHER_ROWS, MESSY_TEXT_ROWS))
    missing_speed, impossible, missing_weather, messy, untouched = np.split(picked, cuts)

    dirty.loc[missing_speed, "olculen_hiz"] = pd.NA
    for position, row in enumerate(impossible):
        dirty.loc[row, "olculen_hiz"] = IMPOSSIBLE_SPEEDS[position % len(IMPOSSIBLE_SPEEDS)]
    dirty["hava"] = dirty["hava"].astype("object")
    dirty.loc[missing_weather, "hava"] = np.nan
    for position, row in enumerate(messy):
        column = "arac_tipi" if position % 2 == 0 else "yol_tipi"
        dirty.loc[row, column] = _messy(dirty.loc[row, column], position)

    copies = dirty.loc[untouched[:DUPLICATE_ROWS]]
    combined = pd.concat([dirty, copies], ignore_index=True)
    return combined.sample(frac=1, random_state=seed).reset_index(drop=True)


def build_csv(path: pathlib.Path = DEFAULT_CSV, n_rows: int = DEFAULT_ROWS, seed: int = DEFAULT_SEED) -> pathlib.Path:
    """Write the damaged table to path and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    add_dirty_rows(generate_records(n_rows, seed), seed).to_csv(path, index=False, encoding="utf-8")
    return path


if __name__ == "__main__":
    print(f"wrote {build_csv()}")
