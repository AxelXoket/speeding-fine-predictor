"""Behaviour of the 2026 speeding rules, checked against cases read from the law and the regulation.

Expected values come from 7574 sayılı Kanun madde 14 (KTK 51/2, 51/3, 51/4), Karayolları Trafik
Yönetmeliği madde 100 and Kabahatler Kanunu 17/6, not from the module under test.
"""
import unittest

import numpy

from speeding_fine import rules

# (vehicle, road, measured km/h, expected fine TL, expected licence suspension days)
URBAN_CAR_CASES = [
    (55, 0, 0), (56, 2_000, 0), (60, 2_000, 0), (61, 4_000, 0), (65, 4_000, 0), (66, 6_000, 0),
    (70, 6_000, 0), (71, 8_000, 0), (75, 8_000, 0), (76, 12_000, 0), (85, 12_000, 0), (86, 15_000, 0),
    (95, 15_000, 0), (96, 20_000, 30), (105, 20_000, 30), (106, 25_000, 60), (115, 25_000, 60),
    (116, 30_000, 90), (200, 30_000, 90),
]
RURAL_CAR_CASES = [  # otomobil on an otoyol, limit 120
    (130, 0, 0), (131, 2_000, 0), (135, 2_000, 0), (136, 4_000, 0), (140, 4_000, 0), (141, 6_000, 0),
    (145, 6_000, 0), (146, 8_000, 0), (150, 8_000, 0), (151, 12_000, 0), (160, 12_000, 0), (161, 15_000, 0),
    (170, 15_000, 0), (171, 20_000, 30), (180, 20_000, 30), (181, 25_000, 60), (190, 25_000, 60),
    (191, 30_000, 90), (300, 30_000, 90),
]
# One case per vehicle and road: the first fined speed and the last free speed, from the regulation table.
FIRST_FINED_SPEED = {
    ("otomobil", "yerlesim_yeri"): 56, ("otomobil", "sehirlerarasi_cift_yonlu"): 101,
    ("otomobil", "bolunmus_yol"): 121, ("otomobil", "otoyol"): 131,
    ("panelvan", "yerlesim_yeri"): 56, ("panelvan", "sehirlerarasi_cift_yonlu"): 96,
    ("panelvan", "bolunmus_yol"): 111, ("panelvan", "otoyol"): 121,
    ("kamyonet", "yerlesim_yeri"): 56, ("kamyonet", "sehirlerarasi_cift_yonlu"): 91,
    ("kamyonet", "bolunmus_yol"): 96, ("kamyonet", "otoyol"): 106,
    ("motosiklet", "yerlesim_yeri"): 56, ("motosiklet", "sehirlerarasi_cift_yonlu"): 91,
    ("motosiklet", "bolunmus_yol"): 101, ("motosiklet", "otoyol"): 111,
    ("minibus", "yerlesim_yeri"): 56, ("minibus", "sehirlerarasi_cift_yonlu"): 91,
    ("minibus", "bolunmus_yol"): 101, ("minibus", "otoyol"): 111,
    ("otobus", "yerlesim_yeri"): 56, ("otobus", "sehirlerarasi_cift_yonlu"): 91,
    ("otobus", "bolunmus_yol"): 101, ("otobus", "otoyol"): 111,
    ("kamyon", "yerlesim_yeri"): 56, ("kamyon", "sehirlerarasi_cift_yonlu"): 91,
    ("kamyon", "bolunmus_yol"): 96, ("kamyon", "otoyol"): 101,
}


class FineTiersFollowTheLaw(unittest.TestCase):
    def check_cases(self, road, cases):
        for speed, fine, days in cases:
            with self.subTest(road=road, speed=speed):
                penalty = rules.penalty_for("otomobil", road, speed)
                self.assertEqual((penalty.fine_tl, penalty.suspension_days), (fine, days))

    def test_every_urban_tier_boundary_from_both_sides(self):
        self.check_cases("yerlesim_yeri", URBAN_CAR_CASES)

    def test_every_rural_tier_boundary_from_both_sides(self):
        self.check_cases("otoyol", RURAL_CAR_CASES)

    def test_all_three_rural_road_types_use_the_rural_tiers(self):
        # Same excess (+11 km/h) is the first rural tier on every road outside a settlement.
        for road, limit in (("sehirlerarasi_cift_yonlu", 90), ("bolunmus_yol", 110), ("otoyol", 120)):
            with self.subTest(road=road):
                self.assertEqual(rules.penalty_for("otomobil", road, limit + 10).fine_tl, 0)
                self.assertEqual(rules.penalty_for("otomobil", road, limit + 11).fine_tl, 2_000)
                self.assertEqual(rules.penalty_for("otomobil", road, limit + 15).fine_tl, 2_000)
                self.assertEqual(rules.penalty_for("otomobil", road, limit + 16).fine_tl, 4_000)

    def test_penalty_reports_limit_and_excess(self):
        penalty = rules.penalty_for("kamyon", "otoyol", 117)
        self.assertEqual((penalty.speed_limit, penalty.excess_kmh), (90, 27))
        self.assertEqual(penalty.fine_tl, 8_000)
        under = rules.penalty_for("otomobil", "yerlesim_yeri", 30)
        self.assertEqual((under.excess_kmh, under.fine_tl, under.suspension_days), (-20, 0, 0))


class SpeedLimitsFollowTheRegulation(unittest.TestCase):
    def test_first_fined_and_last_free_speed_for_every_vehicle_and_road(self):
        self.assertEqual(set(FIRST_FINED_SPEED), {(v, r) for v in rules.VEHICLE_TYPES for r in rules.ROAD_TYPES})
        for (vehicle, road), first_fined in FIRST_FINED_SPEED.items():
            with self.subTest(vehicle=vehicle, road=road):
                self.assertEqual(rules.penalty_for(vehicle, road, first_fined - 1).fine_tl, 0)
                self.assertEqual(rules.penalty_for(vehicle, road, first_fined).fine_tl, 2_000)

    def test_speed_limit_examples(self):
        self.assertEqual(rules.speed_limit("otomobil", "otoyol"), 120)
        self.assertEqual(rules.speed_limit("kamyon", "otoyol"), 90)
        self.assertEqual(rules.speed_limit("panelvan", "bolunmus_yol"), 100)
        self.assertEqual(rules.speed_limit("kamyonet", "sehirlerarasi_cift_yonlu"), 80)

    def test_every_vehicle_and_road_has_a_turkish_label(self):
        self.assertEqual(tuple(rules.VEHICLE_LABELS), rules.VEHICLE_TYPES)
        self.assertEqual(tuple(rules.ROAD_LABELS), rules.ROAD_TYPES)
        self.assertEqual(rules.VEHICLE_LABELS["minibus"], "Minibüs")
        self.assertEqual(rules.ROAD_LABELS["yerlesim_yeri"], "Yerleşim yeri içi")


class FineGrowsWithSpeed(unittest.TestCase):
    def test_fine_never_drops_as_speed_rises_and_starts_after_tolerance(self):
        first_tier = {"yerlesim_yeri": 6}
        checked = 0
        for vehicle in rules.VEHICLE_TYPES:
            for road in rules.ROAD_TYPES:
                previous = (0, 0)
                for speed in range(1, rules.MAX_PLAUSIBLE_SPEED + 1):
                    penalty = rules.penalty_for(vehicle, road, speed)
                    current = (penalty.fine_tl, penalty.suspension_days)
                    self.assertGreaterEqual(current, previous, (vehicle, road, speed))
                    self.assertEqual(penalty.fine_tl > 0, penalty.excess_kmh >= first_tier.get(road, 11))
                    previous = current
                    checked += 1
        self.assertEqual(checked, len(rules.VEHICLE_TYPES) * len(rules.ROAD_TYPES) * rules.MAX_PLAUSIBLE_SPEED)

    def test_tier_tables_have_no_gaps_and_end_open(self):
        for tiers in (rules.URBAN_TIERS, rules.RURAL_TIERS):
            for lower, upper in zip(tiers, tiers[1:]):
                self.assertEqual(upper.min_excess, lower.max_excess + 1)
                self.assertGreater(upper.fine_tl, lower.fine_tl)
            self.assertIsNone(tiers[-1].max_excess)
            self.assertEqual([t.suspension_days for t in tiers if t.suspension_days], [30, 60, 90])


class InvalidInputIsRejected(unittest.TestCase):
    def test_unknown_vehicle_or_road(self):
        with self.assertRaisesRegex(ValueError, "unknown vehicle 'tir'"):
            rules.penalty_for("tir", "otoyol", 100)
        with self.assertRaisesRegex(ValueError, "unknown road 'Otoyol'"):
            rules.speed_limit("otomobil", "Otoyol")
        with self.assertRaisesRegex(ValueError, "unknown road"):
            rules.tiers_for("koy_yolu")

    def test_speed_must_be_a_whole_number(self):
        for bad in (100.0, "100", None, True, numpy.float64(131.0), numpy.bool_(True)):
            with self.subTest(bad=bad), self.assertRaises(TypeError):
                rules.penalty_for("otomobil", "otoyol", bad)

    def test_numpy_whole_numbers_from_a_table_are_accepted(self):
        penalty = rules.penalty_for("otomobil", "otoyol", numpy.int64(131))
        self.assertEqual((penalty.excess_kmh, penalty.fine_tl), (11, 2_000))
        self.assertIs(type(penalty.excess_kmh), int)
        with self.assertRaises(ValueError):
            rules.penalty_for("otomobil", "otoyol", numpy.int64(0))

    def test_speed_must_be_physically_possible(self):
        for bad in (0, -5, rules.MAX_PLAUSIBLE_SPEED + 1):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                rules.penalty_for("otomobil", "otoyol", bad)
        self.assertEqual(rules.penalty_for("otomobil", "otoyol", 1).fine_tl, 0)
        self.assertEqual(rules.penalty_for("otomobil", "otoyol", rules.MAX_PLAUSIBLE_SPEED).fine_tl, 30_000)


class EarlyPayment(unittest.TestCase):
    def test_quarter_off_within_one_month(self):
        self.assertEqual(rules.early_payment_amount(2_000), 1_500)
        self.assertEqual(rules.early_payment_amount(30_000), 22_500)
        self.assertEqual(rules.early_payment_amount(0), 0)
        self.assertEqual(rules.EARLY_PAYMENT_PERIOD, "1 ay")

    def test_negative_fine_is_rejected(self):
        with self.assertRaises(ValueError):
            rules.early_payment_amount(-1)


if __name__ == "__main__":
    unittest.main()
