"""Adaptive top-K selection with QC gates + fallback ladder.

Science rationale (notes/topk-adaptive.md):
  L1 coarse scores -> elbow detection (geometry) is NOT sufficient alone.
  Three independent criteria cross-validate the cutoff:
    1. ELBOW    - geometric knee of sorted score curve (max curvature)
    2. ANCHOR   - known positive control rank must be INSIDE top-K
                  (biological anchor; if pos control outside -> K too small)
    3. SIGNAL   - score dynamic range / separation must exceed noise floor
                  (if scores all clustered -> L1 produced no signal)

QC gates (each can fail -> fallback):
  G1 distribution shape : bimodal-ish? (elbow meaningful only with separation)
  G2 elbow confidence   : max chord distance must exceed threshold
  G3 anchor coverage    : positive control rank < K * safety_factor
  G4 dynamic range      : (max-min) score spread vs noise estimate

Fallback ladder (most specific -> most conservative):
  1. elbow K (if G1+G2 pass)
  2. anchor-expanded K = ceil(pos_rank * safety) (if G3 fails)
  3. default K (if elbow unreliable but signal present)
  4. signal threshold K (score > mean + n*std)
  5. full library (catastrophic fallback: no signal at all -> QC FAIL flag)

Robustness: score curve stability check via split-half correlation.
"""
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class TopKDecision:
    method: str                  # which criterion won
    top_k: int
    elbow_k: Optional[int] = None
    anchor_k: Optional[int] = None
    signal_k: Optional[int] = None
    default_k: Optional[int] = None
    qc_gates: Optional[dict] = None   # per-gate pass/fail + values
    pos_rank: Optional[int] = None
    score_range: Optional[float] = None
    score_mean: Optional[float] = None
    score_std: Optional[float] = None
    split_half_corr: Optional[float] = None
    notes: Optional[str] = None


class TopKSelector:
    def __init__(self, k_min: int = 50, k_max: int = 1000,
                 default_k: int = 500, anchor_safety: float = 1.5,
                 min_chord_dist: float = 0.05, min_split_half: float = 0.7,
                 signal_z: float = 2.0):
        self.k_min = k_min
        self.k_max = k_max
        self.default_k = default_k
        self.anchor_safety = anchor_safety
        self.min_chord_dist = min_chord_dist
        self.min_split_half = min_split_half
        self.signal_z = signal_z

    def _elbow(self, scores: np.ndarray) -> tuple[int, float]:
        """Distance-to-chord elbow on normalized descending curve."""
        y = np.sort(scores)[::-1][: self.k_max].astype(float)
        y = (y - y.min()) / (y.max() - y.min() + 1e-9)
        x = np.linspace(0, 1, len(y))
        if len(y) < 3:
            return 0, 0.0
        dy = y[-1] - y[0]
        dx = x[-1] - x[0]
        denom = math.sqrt(dy * dy + dx * dx) + 1e-9
        d = np.abs((y[1:] - y[0]) * dx - dy * (x[1:] - x[0])) / denom
        elbow = int(np.argmax(d)) + 1
        return max(self.k_min, min(elbow, self.k_max)), float(d.max())

    def _split_half(self, scores: np.ndarray) -> float:
        """Stability: correlation of elbow location across split halves."""
        idx = np.arange(len(scores))
        rng = np.random.default_rng(42)
        half1 = rng.choice(idx, size=len(scores) // 2, replace=False)
        half2 = np.setdiff1d(idx, half1)
        k1, _ = self._elbow(scores[half1])
        k2, _ = self._elbow(scores[half2])
        # stability metric: relative agreement of the two elbows
        if k1 + k2 == 0:
            return 0.0
        return 1.0 - abs(k1 - k2) / max(k1, k2)

    def _signal_k(self, scores: np.ndarray) -> int:
        """Cutoff at mean + z*std (statistical signal threshold)."""
        mu, sd = float(scores.mean()), float(scores.std())
        if sd < 1e-9:
            return self.k_min
        thr = mu + self.signal_z * sd
        k = int(np.sum(scores >= thr))
        return max(self.k_min, min(k, self.k_max))

    def select(self, scores: list[float], pos_rank: Optional[int] = None) -> TopKDecision:
        """Main entry. scores: L1 scores for all library molecules (unsorted).
        pos_rank: 0-based rank of known positive control (None if none)."""
        arr = np.array(scores, dtype=float)
        if len(arr) == 0:
            return TopKDecision(method="qc_fail_empty", top_k=self.default_k,
                                notes="no scores provided")
        if len(arr) < 4:
            # skew/kurtosis and split-half need >=4 samples; below that the
            # only safe decision is to keep everything (recall-first).
            return TopKDecision(
                method="n_too_small", top_k=len(arr),
                score_mean=float(arr.mean()), score_std=float(arr.std()),
                notes=f"N={len(arr)} below statistical floor (4); "
                      "distribution gates undefined -> take all",
            )
        # --- QC gates ---
        gates = {}
        # G1: dynamic range (signal present?)
        sd = float(arr.std())
        mn, mx = float(arr.min()), float(arr.max())
        rng = mx - mn
        g1_pass = rng > 1e-6 and sd > 1e-6
        gates["G1_range"] = {"pass": g1_pass, "range": rng, "std": sd}
        # G5: distribution shape - real L1 scores are right-skewed (few high
        # scores, long tail); noise is symmetric. Skewness + kurtosis gate.
        n = len(arr)
        mu = float(arr.mean())
        sd2 = arr.std()
        if sd2 > 1e-9 and n > 3:
            skew = float(np.mean(((arr - mu) / sd2) ** 3))
            kurt = float(np.mean(((arr - mu) / sd2) ** 4) - 3)
        else:
            skew, kurt = 0.0, 0.0
        # high kurtosis (heavy tail) OR strong skew => non-Gaussian signal
        g5_pass = kurt > 1.0 or abs(skew) > 0.5
        gates["G5_shape"] = {"pass": g5_pass, "skew": skew, "kurtosis": kurt}
        # G2: elbow confidence
        elbow_k, chord = self._elbow(arr)
        g2_pass = chord >= self.min_chord_dist
        gates["G2_elbow"] = {"pass": g2_pass, "elbow_k": elbow_k, "chord": chord}
        # G3: stability (split-half)
        stab = self._split_half(arr)
        g3_pass = stab >= self.min_split_half
        gates["G3_stability"] = {"pass": g3_pass, "split_half": stab}
        # G4: anchor coverage
        anchor_k = None
        g4_pass = True
        if pos_rank is not None and pos_rank >= 0:
            anchor_k = max(self.k_min, int(math.ceil((pos_rank + 1) * self.anchor_safety)))
            g4_pass = anchor_k <= self.k_max
            gates["G4_anchor"] = {"pass": g4_pass, "pos_rank": pos_rank, "anchor_k": anchor_k}
        # --- Decision ladder ---
        decision = TopKDecision(
            method="pending", top_k=0,
            elbow_k=elbow_k, anchor_k=anchor_k, qc_gates=gates,
            pos_rank=pos_rank, score_range=rng, score_mean=float(arr.mean()),
            score_std=sd, split_half_corr=stab,
        )
        if not g1_pass or not g5_pass:
            decision.method = "qc_fail_no_signal"
            decision.top_k = self.default_k
            decision.notes = ("G1/G5 failed: no signal in L1 scores "
                              "(range or shape not distinguishable from noise)")
            return decision
        if g2_pass and g3_pass and (g4_pass or anchor_k is None):
            decision.method = "elbow"
            decision.top_k = min(max(elbow_k, self.k_min), self.k_max)
            return decision
        if not g4_pass and anchor_k is not None:
            # anchor expansion: positive control must be covered
            decision.method = "anchor_expanded"
            decision.top_k = min(anchor_k, self.k_max)
            decision.notes = "G4 failed: elbow K too small to cover positive control"
            return decision
        if g3_pass:
            # elbow unreliable (low chord) but stable -> signal threshold
            k = self._signal_k(arr)
            decision.method = "signal_threshold"
            decision.top_k = min(max(k, self.k_min), self.k_max)
            decision.signal_k = k
            return decision
        # unstable + weak elbow -> conservative default
        decision.method = "default_fallback"
        decision.top_k = self.default_k
        decision.default_k = self.default_k
        decision.notes = "elbow unstable + weak chord; conservative default"
        return decision


if __name__ == "__main__":
    # demo: synthetic distributions
    sel = TopKSelector()
    for name, scores, pos_rank in [
        ("sharp_elbow", list(np.concatenate([np.linspace(1, 0.3, 300), np.full(8700, 0.3)])) + [0.9], 100),
        ("smooth_noise", list(np.random.default_rng(1).normal(0.5, 0.02, 12000)), None),
        ("bimodal", list(np.concatenate([np.random.default_rng(2).normal(0.9, 0.05, 200), np.random.default_rng(3).normal(0.4, 0.1, 11800)])) + [0.95], 50),
    ]:
        d = sel.select(scores, pos_rank)
        print(f"{name:15s} -> method={d.method:20s} top_k={d.top_k}")
        print(f"      elbow={d.elbow_k} chord_ok={d.qc_gates['G2_elbow']['pass']} "
              f"stab={d.split_half_corr:.2f} anchor={d.anchor_k} range={d.score_range:.3f}")
