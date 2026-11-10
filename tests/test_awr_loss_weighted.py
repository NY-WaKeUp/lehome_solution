"""Unit tests for AWR loss-weighted ablation helpers (no heavy model imports)."""

from __future__ import annotations

import numpy as np

from lehome_solution.training.loss_weights import normalized_exp_advantage_weights


def test_normalized_exp_advantage_weights():
    adv = np.array([-2.0, 0.0, 2.0], dtype=np.float32)
    w = np.asarray(normalized_exp_advantage_weights(adv))
    assert np.isclose(w.sum(), 1.0)
    assert w[-1] > w[0]
    assert np.isclose(w[-1] / w[0], np.exp(4.0), rtol=1e-5)
