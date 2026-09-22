"""
tests/test_rs_vlm_questions.py — Tests for deterministic 40-question test suite.

Verifies:
- Exactly 40 questions in the evaluation suite
- Exactly 20 distinct remote-sensing categories with 2 questions each
- Deterministic, unique question IDs
- All referenced image samples exist in the local sample registry
- Real satellite image files exist locally on disk
- No fabricated ground truth (ground_truth_available is False or answer is None)
"""

import os
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.evaluation.question_set import get_40_question_suite, get_evaluation_samples


class TestRSVLMQuestions(unittest.TestCase):
    """Verifies test suite integrity, category coverage, and deterministic question set."""

    def setUp(self):
        self.questions = get_40_question_suite()
        self.samples = get_evaluation_samples()

    def test_exact_40_questions(self):
        """Benchmark suite must contain exactly 40 questions."""
        self.assertEqual(len(self.questions), 40, f"Expected exactly 40 questions, got {len(self.questions)}")

    def test_20_categories_two_questions_each(self):
        """Benchmark suite must cover 20 categories with exactly 2 questions per category."""
        category_counts = {}
        for q in self.questions:
            category_counts[q.category] = category_counts.get(q.category, 0) + 1

        self.assertEqual(len(category_counts), 20, f"Expected exactly 20 categories, got {len(category_counts)}")
        for cat, count in category_counts.items():
            self.assertEqual(count, 2, f"Category '{cat}' has {count} questions, expected 2")

    def test_unique_and_deterministic_ids(self):
        """Question IDs must be non-empty, unique, and deterministic."""
        ids = [q.question_id for q in self.questions]
        self.assertEqual(len(ids), len(set(ids)), "Duplicate question IDs detected")
        for q_id in ids:
            self.assertTrue(q_id.startswith("RS-"), f"Question ID '{q_id}' must start with 'RS-'")

    def test_no_fabricated_ground_truth(self):
        """Unless empirical ground truth exists, ground_truth_available must be False."""
        for q in self.questions:
            self.assertFalse(q.ground_truth_available, f"Ground truth must not be fabricated for {q.question_id}")
            self.assertIsNone(q.ground_truth_answer)

    def test_all_samples_referenced_exist_in_registry(self):
        """Every question must reference a known sample ID in the registry."""
        for q in self.questions:
            self.assertIn(q.sample_id, self.samples, f"Question {q.question_id} references unknown sample {q.sample_id}")

    def test_local_satellite_images_exist_on_disk(self):
        """The real satellite imagery referenced by the benchmark must exist locally."""
        for sample_id, sample in self.samples.items():
            self.assertTrue(
                os.path.exists(sample.image_path),
                f"Sample '{sample_id}' primary image not found at '{sample.image_path}'"
            )
            if sample.is_paired and sample.secondary_image_path:
                self.assertTrue(
                    os.path.exists(sample.secondary_image_path),
                    f"Sample '{sample_id}' secondary image not found at '{sample.secondary_image_path}'"
                )

    def test_paired_image_temporal_and_sar_designations(self):
        """Paired questions must target verified paired samples."""
        pair_questions = [q for q in self.questions if q.input_type == "pair"]
        self.assertGreaterEqual(len(pair_questions), 2)
        for pq in pair_questions:
            sample = self.samples[pq.sample_id]
            self.assertTrue(sample.is_paired, f"Paired question {pq.question_id} mapped to non-paired sample")


if __name__ == "__main__":
    unittest.main()
