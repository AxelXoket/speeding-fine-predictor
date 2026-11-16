"""Behaviour of the synthetic radar record generator and of the deliberate damage applied to it."""
import pathlib
import tempfile
import unittest

import pandas as pd

from speeding_fine import dataset, rules

CLEAN = dataset.generate_records()
DIRTY = dataset.add_dirty_rows(CLEAN)


def normalised(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in ("arac_tipi", "yol_tipi"):
        frame[column] = frame[column].str.strip().str.lower()
    return frame


class GeneratedRecords(unittest.TestCase):
    def test_shape_and_columns(self):
        self.assertEqual(len(CLEAN), dataset.DEFAULT_ROWS)
        self.assertEqual(tuple(CLEAN.columns), dataset.COLUMNS)
        self.assertEqual(CLEAN["kayit_no"].tolist(), list(range(1, dataset.DEFAULT_ROWS + 1)))
        self.assertEqual(int(CLEAN.isnull().sum().sum()), 0)

    def test_same_seed_same_table_other_seed_other_table(self):
        pd.testing.assert_frame_equal(dataset.generate_records(50, seed=7), dataset.generate_records(50, seed=7))
        self.assertFalse(dataset.generate_records(50, seed=7).equals(dataset.generate_records(50, seed=8)))

    def test_every_label_and_limit_matches_the_rules(self):
        for row in CLEAN.itertuples(index=False):
            penalty = rules.penalty_for(row.arac_tipi, row.yol_tipi, int(row.olculen_hiz))
            self.assertEqual(row.hiz_siniri, penalty.speed_limit)
            self.assertEqual((row.ceza_tutari_tl, row.ehliyet_geri_alma_gun), (penalty.fine_tl, penalty.suspension_days))

    def test_values_stay_in_their_domains(self):
        self.assertTrue(set(CLEAN["arac_tipi"]) <= set(rules.VEHICLE_TYPES))
        self.assertTrue(set(CLEAN["yol_tipi"]) <= set(rules.ROAD_TYPES))
        self.assertTrue(set(CLEAN["hava"]) <= set(dataset.WEATHER_TYPES))
        self.assertEqual(tuple(dataset.WEATHER_LABELS), dataset.WEATHER_TYPES)
        self.assertEqual(dataset.WEATHER_LABELS["karli"], "Karlı")
        self.assertTrue(CLEAN["saat"].between(0, 23).all())
        self.assertTrue(CLEAN["olculen_hiz"].between(1, rules.MAX_PLAUSIBLE_SPEED).all())

    def test_every_fine_tier_and_both_kinds_of_free_rows_appear_inside_and_outside_settlements(self):
        excess = CLEAN["olculen_hiz"] - CLEAN["hiz_siniri"]
        for urban, tiers in ((True, rules.URBAN_TIERS), (False, rules.RURAL_TIERS)):
            part = CLEAN[(CLEAN["yol_tipi"] == rules.URBAN_ROAD) == urban]
            part_excess = excess[part.index]
            with self.subTest(urban=urban):
                self.assertEqual(set(part["ceza_tutari_tl"]), {0} | {t.fine_tl for t in tiers})
                self.assertTrue((part_excess <= 0).any())
                self.assertTrue(((part_excess > 0) & (part["ceza_tutari_tl"] == 0)).any())
        self.assertEqual(set(CLEAN["arac_tipi"]), set(rules.VEHICLE_TYPES))
        self.assertEqual(set(CLEAN["yol_tipi"]), set(rules.ROAD_TYPES))

    def test_open_top_tier_stays_within_its_sampling_width(self):
        excess = CLEAN["olculen_hiz"] - CLEAN["hiz_siniri"]
        for road in rules.ROAD_TYPES:
            top = rules.tiers_for(road)[-1]
            rows = excess[CLEAN["yol_tipi"] == road]
            self.assertLessEqual(rows.max(), top.min_excess + dataset.OPEN_TIER_WIDTH - 1)
            self.assertGreaterEqual(rows.min(), dataset.UNDER_LIMIT_MIN_EXCESS)

    def test_rejects_empty_table(self):
        with self.assertRaises(ValueError):
            dataset.generate_records(0)


class DeliberateDamage(unittest.TestCase):
    def test_exact_number_of_each_problem(self):
        speed = DIRTY["olculen_hiz"]
        self.assertEqual(int(speed.isnull().sum()), dataset.MISSING_SPEED_ROWS)
        impossible = speed.notnull() & ~speed.between(1, rules.MAX_PLAUSIBLE_SPEED)
        self.assertEqual(int(impossible.sum()), dataset.IMPOSSIBLE_SPEED_ROWS)
        self.assertEqual(int(DIRTY["hava"].isnull().sum()), dataset.MISSING_WEATHER_ROWS)
        messy = (DIRTY["arac_tipi"] != DIRTY["arac_tipi"].str.strip().str.lower()) | (
            DIRTY["yol_tipi"] != DIRTY["yol_tipi"].str.strip().str.lower()
        )
        self.assertEqual(int(messy.sum()), dataset.MESSY_TEXT_ROWS)
        for column in ("arac_tipi", "yol_tipi"):
            self.assertTrue((DIRTY[column] != DIRTY[column].str.strip().str.lower()).any(), column)
        self.assertEqual(int(DIRTY.duplicated().sum()), dataset.DUPLICATE_ROWS)
        self.assertEqual(len(DIRTY), dataset.DEFAULT_ROWS + dataset.DUPLICATE_ROWS)

    def test_each_damaged_row_has_one_problem_and_duplicates_copy_clean_rows(self):
        deduped = DIRTY.drop_duplicates().set_index("kayit_no").sort_index()
        reference = CLEAN.set_index("kayit_no")
        changed = (deduped.astype(str) != reference.astype(str)).sum(axis=1)
        self.assertEqual(int((changed > 1).sum()), 0)
        damaged_total = (
            dataset.MISSING_SPEED_ROWS + dataset.IMPOSSIBLE_SPEED_ROWS + dataset.MISSING_WEATHER_ROWS + dataset.MESSY_TEXT_ROWS
        )
        self.assertEqual(int((changed == 1).sum()), damaged_total)
        duplicated_ids = DIRTY[DIRTY.duplicated()]["kayit_no"]
        self.assertTrue((changed[duplicated_ids] == 0).all())

    def test_normalising_text_restores_every_messy_value(self):
        restored = normalised(DIRTY.drop_duplicates()).set_index("kayit_no").sort_index()
        reference = CLEAN.set_index("kayit_no")
        for column in ("arac_tipi", "yol_tipi"):
            self.assertEqual(restored[column].tolist(), reference[column].tolist())

    def test_input_table_is_not_modified_and_impossible_speeds_are_the_documented_ones(self):
        before = CLEAN.copy()
        dirty = dataset.add_dirty_rows(CLEAN)
        pd.testing.assert_frame_equal(CLEAN, before)
        speed = dirty["olculen_hiz"].dropna()
        self.assertEqual(sorted(speed[~speed.between(1, rules.MAX_PLAUSIBLE_SPEED)].tolist()), sorted(dataset.IMPOSSIBLE_SPEEDS))

    def test_any_index_on_the_input_gives_the_same_damage(self):
        shifted = CLEAN.set_index(CLEAN.index + 1000)
        pd.testing.assert_frame_equal(dataset.add_dirty_rows(shifted), DIRTY)

    def test_seed_must_be_a_whole_number(self):
        for bad in (None, 1.5, "7", True):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    dataset.generate_records(50, seed=bad)
                with self.assertRaises(TypeError):
                    dataset.add_dirty_rows(CLEAN, seed=bad)

    def test_too_small_table_is_rejected(self):
        needed = (
            dataset.MISSING_SPEED_ROWS + dataset.IMPOSSIBLE_SPEED_ROWS + dataset.MISSING_WEATHER_ROWS
            + dataset.MESSY_TEXT_ROWS + dataset.DUPLICATE_ROWS
        )
        with self.assertRaises(ValueError):
            dataset.add_dirty_rows(dataset.generate_records(needed - 1))
        self.assertEqual(len(dataset.add_dirty_rows(dataset.generate_records(needed))), needed + dataset.DUPLICATE_ROWS)


class CsvFile(unittest.TestCase):
    def test_committed_csv_is_exactly_what_the_generator_writes(self):
        with tempfile.TemporaryDirectory() as folder:
            fresh = dataset.build_csv(pathlib.Path(folder) / "records.csv")
            self.assertEqual(fresh.read_bytes(), dataset.DEFAULT_CSV.read_bytes())

    def test_the_csv_has_lf_line_endings_on_every_system(self):
        with tempfile.TemporaryDirectory() as folder:
            fresh = dataset.build_csv(pathlib.Path(folder) / "records.csv").read_bytes()
        self.assertGreater(fresh.count(b"\n"), len(DIRTY))
        self.assertNotIn(b"\r", fresh)
        self.assertNotIn(b"\r", dataset.DEFAULT_CSV.read_bytes())

    def test_csv_round_trip_keeps_values_and_blanks(self):
        loaded = pd.read_csv(dataset.DEFAULT_CSV)
        self.assertEqual(tuple(loaded.columns), dataset.COLUMNS)
        self.assertEqual(len(loaded), len(DIRTY))
        self.assertEqual(int(loaded["olculen_hiz"].isnull().sum()), dataset.MISSING_SPEED_ROWS)
        self.assertEqual(int(loaded["hava"].isnull().sum()), dataset.MISSING_WEATHER_ROWS)
        speeds = loaded["olculen_hiz"].dropna()
        self.assertTrue((speeds == speeds.round()).all())


if __name__ == "__main__":
    unittest.main()
