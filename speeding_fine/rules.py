"""Turkish speeding rules valid in 2026: default speed limits and the fine for exceeding them.

Fine tiers: 2918 sayılı KTK madde 51/2 as rewritten by 7574 sayılı Kanun madde 14
(Resmî Gazete 27.02.2026, sayı 33181, in force on publication). The amounts are fixed in the law text.
Reading KTK Ek Madde 3 with Kabahatler Kanunu 17/7, the yearly revaluation first applies on 01.01.2027;
no official notice confirming that date was found.
Licence suspension: KTK 51/3 and 51/4, "her seferinde" 30, 60 and 90 days for tiers 7, 8 and 9.
Speed limits: Karayolları Trafik Yönetmeliği madde 100, defaults for vehicles without a trailer when no other
limit is signposted. "motosiklet" is class L3; L4, L5 and L7 have lower limits and are not modelled.
Early payment: a traffic fine is due within one month of notification (KTK 115/1); paid within that period it is
25% off (Kabahatler Kanunu 17/6).
"""
import numbers
from dataclasses import dataclass

URBAN_ROAD = "yerlesim_yeri"
ROAD_TYPES = (URBAN_ROAD, "sehirlerarasi_cift_yonlu", "bolunmus_yol", "otoyol")
ROAD_LABELS = {
    URBAN_ROAD: "Yerleşim yeri içi",
    "sehirlerarasi_cift_yonlu": "Şehirlerarası çift yönlü yol",
    "bolunmus_yol": "Bölünmüş yol",
    "otoyol": "Otoyol",
}

# Limits in ROAD_TYPES order (km/h).
SPEED_LIMITS = {
    "otomobil": (50, 90, 110, 120),
    "panelvan": (50, 85, 100, 110),
    "kamyonet": (50, 80, 85, 95),
    "motosiklet": (50, 80, 90, 100),
    "minibus": (50, 80, 90, 100),
    "otobus": (50, 80, 90, 100),
    "kamyon": (50, 80, 85, 90),
}
VEHICLE_TYPES = tuple(SPEED_LIMITS)
VEHICLE_LABELS = {
    "otomobil": "Otomobil",
    "panelvan": "Panelvan",
    "kamyonet": "Kamyonet",
    "motosiklet": "Motosiklet",
    "minibus": "Minibüs",
    "otobus": "Otobüs",
    "kamyon": "Kamyon",
}

# Karayolları Trafik Yönetmeliği madde 100/2: vehicles that must carry a speed limiter (M2/M3 minibüs and otobüs,
# N2/N3 kamyon) have it set to these speeds on every road. The data generator ignores the limiter on purpose (README).
SPEED_LIMITER = {"minibus": 110, "otobus": 110, "kamyon": 99}

MAX_PLAUSIBLE_SPEED = 300
EARLY_PAYMENT_DISCOUNT = 0.25
EARLY_PAYMENT_PERIOD = "1 ay"


@dataclass(frozen=True)
class FineTier:
    min_excess: int
    max_excess: int | None  # inclusive; None means no upper bound
    fine_tl: int
    suspension_days: int


URBAN_TIERS = (
    FineTier(6, 10, 2_000, 0),
    FineTier(11, 15, 4_000, 0),
    FineTier(16, 20, 6_000, 0),
    FineTier(21, 25, 8_000, 0),
    FineTier(26, 35, 12_000, 0),
    FineTier(36, 45, 15_000, 0),
    FineTier(46, 55, 20_000, 30),
    FineTier(56, 65, 25_000, 60),
    FineTier(66, None, 30_000, 90),
)
RURAL_TIERS = (
    FineTier(11, 15, 2_000, 0),
    FineTier(16, 20, 4_000, 0),
    FineTier(21, 25, 6_000, 0),
    FineTier(26, 30, 8_000, 0),
    FineTier(31, 40, 12_000, 0),
    FineTier(41, 50, 15_000, 0),
    FineTier(51, 60, 20_000, 30),
    FineTier(61, 70, 25_000, 60),
    FineTier(71, None, 30_000, 90),
)


@dataclass(frozen=True)
class Penalty:
    speed_limit: int
    excess_kmh: int
    fine_tl: int
    suspension_days: int


def _check_choice(value: str, options: tuple[str, ...], kind: str) -> None:
    if value not in options:
        raise ValueError(f"unknown {kind} {value!r}; expected one of {options}")


def speed_limit(vehicle: str, road: str) -> int:
    """Default limit in km/h for this vehicle on this road type."""
    _check_choice(vehicle, VEHICLE_TYPES, "vehicle")
    _check_choice(road, ROAD_TYPES, "road")
    return SPEED_LIMITS[vehicle][ROAD_TYPES.index(road)]


def tiers_for(road: str) -> tuple[FineTier, ...]:
    """Fine tiers that apply on this road type: inside or outside a settlement."""
    _check_choice(road, ROAD_TYPES, "road")
    return URBAN_TIERS if road == URBAN_ROAD else RURAL_TIERS


def penalty_for(vehicle: str, road: str, measured_kmh: int) -> Penalty:
    """Fine and licence suspension for one radar measurement given as whole km/h."""
    if isinstance(measured_kmh, bool) or not isinstance(measured_kmh, numbers.Integral):
        raise TypeError("measured_kmh must be a whole number")
    measured_kmh = int(measured_kmh)
    if not 1 <= measured_kmh <= MAX_PLAUSIBLE_SPEED:
        raise ValueError(f"measured_kmh must be between 1 and {MAX_PLAUSIBLE_SPEED}")
    limit = speed_limit(vehicle, road)
    excess = measured_kmh - limit
    for tier in tiers_for(road):
        if excess >= tier.min_excess and (tier.max_excess is None or excess <= tier.max_excess):
            return Penalty(limit, excess, tier.fine_tl, tier.suspension_days)
    return Penalty(limit, excess, 0, 0)


def early_payment_amount(fine_tl: int) -> int:
    """Amount due when the fine is paid within the early payment period."""
    if fine_tl < 0:
        raise ValueError("fine_tl must not be negative")
    return round(fine_tl * (1 - EARLY_PAYMENT_DISCOUNT))
