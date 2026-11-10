"""
Phase 4 — DAS Data Integrity Tests
====================================
Tests that do NOT require Qwen inference or DAS retraining.
They verify split integrity, metric definitions, and document the
contamination in the old validate_02 script.

Run with:
    cd /Users/amelia/Desktop/llmneuroscientist
    python -m pytest tests/test_das_split_integrity.py -v

All tests use only saved JSON/NPY files and pure NumPy.  No model loading.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
TRIALS_PATH = ROOT / "data/exp6_full/trials.jsonl"
R_PATH      = ROOT / "results/exp7h/R_choice_tendency_L27_k16.npy"
DAS_RES     = ROOT / "results/exp7h/das_results.json"
DAS_EVAL    = ROOT / "results/exp7h/das_evaluation.json"
VAL_RESULTS = ROOT / "VALIDATION_EXP7_V2/VALIDATION_RESULTS.md"
V2_SUMMARY  = ROOT / "VALIDATION_EXP7_V2/results/confirmatory_run/summary.json"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_rows():
    if not TRIALS_PATH.exists():
        pytest.skip(f"trials.jsonl not found: {TRIALS_PATH}")
    return [json.loads(l) for l in TRIALS_PATH.read_text().splitlines() if l]


def das_split(rows, n_train=1200):
    """Exact copy of exp7h_das.episode_aware_split logic."""
    n = len(rows)
    train = np.zeros(n, dtype=bool)
    train[:n_train] = True
    return train, ~train


def old_validate02_split(rows, seed=42, n_train=1200):
    """
    Replicates the episode_split() used in results/validation/validate_02_random_iia.py.
    Uses seed=42 random permutation of episode IDs, selecting first n_train/30 as training.
    """
    eps = sorted(set(r["episode_id"] for r in rows))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(eps))
    n_eps_train = int(n_train / 30)   # 40 training episodes
    train_eps = set(eps[i] for i in perm[:n_eps_train])
    tr = np.array([r["episode_id"] in train_eps for r in rows])
    return tr, ~tr


# ---------------------------------------------------------------------------
# TEST GROUP A — DAS row-index split integrity
# ---------------------------------------------------------------------------

class TestDASSplitIntegrity:

    def test_total_rows(self):
        """Dataset should contain exactly 1500 rows (50 eps × 30 trials)."""
        rows = load_rows()
        assert len(rows) == 1500, f"Expected 1500 rows, got {len(rows)}"

    def test_no_episode_overlap(self):
        """DAS train and val episode sets must be disjoint."""
        rows = load_rows()
        train_mask, val_mask = das_split(rows)
        train_eps = set(r["episode_id"] for r, t in zip(rows, train_mask) if t)
        val_eps   = set(r["episode_id"] for r, v in zip(rows, val_mask)   if v)
        overlap = train_eps & val_eps
        assert overlap == set(), f"Episode overlap found: {overlap}"

    def test_train_episode_count(self):
        """Training set must contain exactly 40 episodes."""
        rows = load_rows()
        train_mask, _ = das_split(rows)
        train_eps = set(r["episode_id"] for r, t in zip(rows, train_mask) if t)
        assert len(train_eps) == 40, f"Expected 40 train episodes, got {len(train_eps)}"

    def test_val_episode_count(self):
        """Validation set must contain exactly 10 episodes."""
        rows = load_rows()
        _, val_mask = das_split(rows)
        val_eps = set(r["episode_id"] for r, v in zip(rows, val_mask) if v)
        assert len(val_eps) == 10, f"Expected 10 val episodes, got {len(val_eps)}"

    def test_train_row_count(self):
        """Training mask must cover exactly 1200 rows."""
        rows = load_rows()
        train_mask, _ = das_split(rows)
        assert int(train_mask.sum()) == 1200, f"Expected 1200 train rows, got {train_mask.sum()}"

    def test_val_row_count(self):
        """Validation mask must cover exactly 300 rows."""
        rows = load_rows()
        _, val_mask = das_split(rows)
        assert int(val_mask.sum()) == 300, f"Expected 300 val rows, got {val_mask.sum()}"

    def test_train_episodes_are_first_40(self):
        """Training episodes must correspond to seq_0000–seq_0039 (first 40 sequential)."""
        rows = load_rows()
        train_mask, _ = das_split(rows)
        train_eps = sorted(set(r["episode_id"] for r, t in zip(rows, train_mask) if t))
        expected = [f"seq_{i:04d}" for i in range(40)]
        assert train_eps == expected, f"Unexpected training episodes: {train_eps[:5]}..."

    def test_val_episodes_are_last_10(self):
        """Validation episodes must correspond to seq_0040–seq_0049."""
        rows = load_rows()
        _, val_mask = das_split(rows)
        val_eps = sorted(set(r["episode_id"] for r, v in zip(rows, val_mask) if v))
        expected = [f"seq_{i:04d}" for i in range(40, 50)]
        assert val_eps == expected, f"Unexpected val episodes: {val_eps}"

    def test_row_boundary(self):
        """Row 1199 must be seq_0039, row 1200 must be seq_0040."""
        rows = load_rows()
        assert rows[1199]["episode_id"] == "seq_0039", (
            f"Row 1199 episode: {rows[1199]['episode_id']}"
        )
        assert rows[1200]["episode_id"] == "seq_0040", (
            f"Row 1200 episode: {rows[1200]['episode_id']}"
        )


# ---------------------------------------------------------------------------
# TEST GROUP B — Contamination documentation (validate_02)
# ---------------------------------------------------------------------------

class TestOldValidateContamination:
    """
    These tests DOCUMENT the contamination in validate_02_random_iia.py.
    They are expected to show 9/10 contamination — that is the confirmed bug.
    """

    def test_contamination_confirmed(self):
        """
        The old validate_02 split (seed=42 random shuffle) places 9/10
        'validation' episodes inside DAS training rows.
        This is the source of the contaminated IIA=0.995 value.
        """
        rows = load_rows()
        das_train_mask, _ = das_split(rows)
        das_train_eps = set(r["episode_id"] for r, t in zip(rows, das_train_mask) if t)

        old_tr_mask, old_val_mask = old_validate02_split(rows, seed=42)
        old_val_eps = set(r["episode_id"] for r, v in zip(rows, old_val_mask) if v)

        overlap = old_val_eps & das_train_eps
        # Assert the contamination exists (this is the DOCUMENTED BUG)
        assert len(overlap) == 9, (
            f"Expected 9/10 contaminated episodes, got {len(overlap)}/10.\n"
            f"Contaminated: {sorted(overlap)}"
        )

    def test_current_split_has_no_contamination(self):
        """
        The current exp7h_das.py row-index split has ZERO contamination.
        Train and val episodes are completely disjoint.
        """
        rows = load_rows()
        das_train_mask, das_val_mask = das_split(rows)
        das_train_eps = set(r["episode_id"] for r, t in zip(rows, das_train_mask) if t)
        das_val_eps   = set(r["episode_id"] for r, v in zip(rows, das_val_mask)   if v)
        overlap = das_train_eps & das_val_eps
        assert len(overlap) == 0, f"Unexpected overlap: {overlap}"


# ---------------------------------------------------------------------------
# TEST GROUP C — R matrix integrity
# ---------------------------------------------------------------------------

class TestRMatrix:

    def test_r_matrix_shape(self):
        """R must be (1024, 16)."""
        if not R_PATH.exists():
            pytest.skip(f"R matrix not found: {R_PATH}")
        R = np.load(R_PATH)
        assert R.shape == (1024, 16), f"Unexpected shape: {R.shape}"

    def test_r_matrix_orthonormality(self):
        """R must satisfy R^T R ≈ I_16 to within 1e-5."""
        if not R_PATH.exists():
            pytest.skip(f"R matrix not found: {R_PATH}")
        R = np.load(R_PATH)
        residual = R.T @ R - np.eye(16)
        max_error = float(np.abs(residual).max())
        assert max_error < 1e-5, f"Orthogonality error {max_error:.2e} exceeds 1e-5"

    def test_r_matrix_finite(self):
        """R must contain no NaN or Inf."""
        if not R_PATH.exists():
            pytest.skip(f"R matrix not found: {R_PATH}")
        R = np.load(R_PATH)
        assert np.isfinite(R).all(), "R matrix contains non-finite values"


# ---------------------------------------------------------------------------
# TEST GROUP D — Saved results consistency
# ---------------------------------------------------------------------------

class TestSavedResults:

    def test_das_results_best_choice_tendency_l27(self):
        """das_results.json must contain choice_tendency L27 k=16 with val IIA ≥ 0.99."""
        if not DAS_RES.exists():
            pytest.skip(f"das_results.json not found: {DAS_RES}")
        data = json.loads(DAS_RES.read_text())
        matching = [r for r in data["runs"]
                    if r["variable"] == "choice_tendency"
                    and r["layer"] == 27
                    and r["k"] == 16]
        assert len(matching) == 1, f"Expected exactly one matching run, got {len(matching)}"
        iia = matching[0]["iia_val"]
        assert iia >= 0.99, f"Expected iia_val >= 0.99 for L27 choice_tendency k=16, got {iia}"

    def test_das_evaluation_heldout_l27(self):
        """das_evaluation.json must contain choice_tendency L27 k=16 with iia_heldout ≥ 0.99."""
        if not DAS_EVAL.exists():
            pytest.skip(f"das_evaluation.json not found: {DAS_EVAL}")
        data = json.loads(DAS_EVAL.read_text())
        matching = [r for r in data
                    if r["variable"] == "choice_tendency"
                    and r["layer"] == 27
                    and r["k"] == 16]
        assert len(matching) == 1, f"Expected exactly one matching entry, got {len(matching)}"
        iia_heldout = matching[0]["iia_heldout"]
        assert iia_heldout >= 0.99, f"Expected iia_heldout >= 0.99, got {iia_heldout}"

    def test_v2_primary_result_range(self):
        """V2 summary.json primary cluster bootstrap mean must be in [0.08, 0.15]."""
        if not V2_SUMMARY.exists():
            pytest.skip(f"V2 summary not found: {V2_SUMMARY}")
        data = json.loads(V2_SUMMARY.read_text())
        bs = data.get("primary_cluster_bootstrap", {})
        mean = bs.get("mean")
        assert mean is not None, "primary_cluster_bootstrap.mean missing from summary"
        assert 0.08 <= mean <= 0.15, f"V2 primary mean {mean:.4f} outside expected range [0.08, 0.15]"

    def test_v2_all_clusters_positive(self):
        """All 10 V2 cluster values must be positive (each cluster DAS > random)."""
        if not V2_SUMMARY.exists():
            pytest.skip(f"V2 summary not found: {V2_SUMMARY}")
        data = json.loads(V2_SUMMARY.read_text())
        clusters = data.get("primary_cluster_bootstrap", {}).get("clusters", [])
        assert len(clusters) == 10, f"Expected 10 clusters, got {len(clusters)}"
        negative = [c for c in clusters if c <= 0]
        assert len(negative) == 0, f"Clusters with non-positive values: {negative}"

    def test_v2_ci_does_not_include_zero(self):
        """V2 95% CI lower bound must be > 0."""
        if not V2_SUMMARY.exists():
            pytest.skip(f"V2 summary not found: {V2_SUMMARY}")
        data = json.loads(V2_SUMMARY.read_text())
        ci = data.get("primary_cluster_bootstrap", {}).get("interval_95_percentile", [])
        assert len(ci) == 2, f"Expected 2-element CI, got {ci}"
        assert ci[0] > 0, f"CI lower bound {ci[0]:.4f} includes zero"


# ---------------------------------------------------------------------------
# TEST GROUP E — Metric definition cross-checks
# ---------------------------------------------------------------------------

class TestMetricDefinitions:
    """Verify that the core.py metrics() function behaves as documented."""

    @pytest.fixture(autouse=True)
    def _import_core(self):
        sys.path.insert(0, str(ROOT / "VALIDATION_EXP7_V2/scripts"))
        try:
            from core import metrics
            self.metrics = metrics
        except ImportError:
            pytest.skip("core.py not importable from VALIDATION_EXP7_V2/scripts")

    def test_no_change_no_flip(self):
        """Zero intervention should produce no argmax flip."""
        result = self.metrics([1.0, 0.0], [1.0, 0.0], offset=0.5, base_target=0.3, source_target=0.7)
        assert result["restricted_ab_argmax_flip"] is False

    def test_direction_correct_when_source_higher(self):
        """When source_target > base_target and intervention shifts A up, DA=True."""
        result = self.metrics([1.0, 0.0], [1.5, 0.0], offset=0.0, base_target=0.3, source_target=0.9)
        assert result["directional_agreement"] is True

    def test_direction_wrong_when_source_higher_but_shift_down(self):
        """When source_target > base_target but intervention shifts A down, DA=False."""
        result = self.metrics([1.0, 0.0], [0.5, 0.0], offset=0.0, base_target=0.3, source_target=0.9)
        assert result["directional_agreement"] is False

    def test_argmax_flip_requires_sign_change(self):
        """Argmax flip only occurs if A−B logit changes sign."""
        result = self.metrics([2.0, 0.0], [0.5, 0.0], offset=0.0, base_target=0.5, source_target=0.5)
        assert result["restricted_ab_argmax_flip"] is False   # still A > B

    def test_argmax_flip_when_sign_changes(self):
        """Confirm argmax flip IS detected when A−B sign changes."""
        result = self.metrics([2.0, 0.0], [-0.5, 0.0], offset=0.0, base_target=0.5, source_target=0.5)
        assert result["restricted_ab_argmax_flip"] is True

    def test_equal_targets_gives_none_da(self):
        """When source_target == base_target, directional_agreement is None."""
        result = self.metrics([1.0, 0.0], [1.5, 0.0], offset=0.0, base_target=0.5, source_target=0.5)
        assert result["directional_agreement"] is None


if __name__ == "__main__":
    import subprocess, sys
    sys.exit(subprocess.call(
        [sys.executable, "-m", "pytest", __file__, "-v"],
        cwd=str(ROOT)
    ))
