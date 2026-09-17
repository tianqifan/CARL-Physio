import unittest

import numpy as np

from carl.data import (
    apply_channel_standardization,
    fit_channel_standardization,
    loso_split,
)


class ProtocolTests(unittest.TestCase):
    def test_rotating_subject_split_is_disjoint(self):
        subject_ids = np.repeat(np.array([2, 3, 4, 5]), 3)
        for fold in range(4):
            train, validation, test, test_subject, validation_subject = loso_split(
                subject_ids, fold
            )
            self.assertFalse(np.any(train & validation))
            self.assertFalse(np.any(train & test))
            self.assertFalse(np.any(validation & test))
            self.assertNotEqual(test_subject, validation_subject)
            self.assertTrue(np.all(train | validation | test))

    def test_normalization_is_fitted_only_on_training_values(self):
        train = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        held_out = np.full((1, 3, 4), 1000.0, dtype=np.float32)
        stats = fit_channel_standardization(train)
        normalized_train = apply_channel_standardization(train, stats)
        normalized_held_out = apply_channel_standardization(held_out, stats)
        np.testing.assert_allclose(normalized_train.mean(axis=(0, 1)), 0.0, atol=1e-6)
        np.testing.assert_allclose(normalized_train.std(axis=(0, 1)), 1.0, atol=1e-6)
        self.assertGreater(float(normalized_held_out.min()), 100.0)


if __name__ == "__main__":
    unittest.main()
