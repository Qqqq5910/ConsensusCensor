"""Classical HT residual and M2 post-stratified estimates; no silent projection."""

from dataclasses import dataclass, field

import numpy as np

from consensuscensor.data.views import Audit, FloatArray, IntArray, Observed, Oracle


@dataclass(frozen=True)
class AuditDesign:
    """Known first-order inclusion probabilities and a single shared released audit."""

    audit: Audit
    pi: FloatArray
    name: str = "fixed-size-SRS-without-replacement"


@dataclass
class ResidualEstimate:
    groups: FloatArray
    status: str = "ok"
    message: str = ""
    diagnostics: dict = field(default_factory=dict)

    @property
    def overall(self) -> FloatArray:
        return self.groups.mean(axis=0)


def unavailable(message: str, status: str = "not_estimable") -> ResidualEstimate:
    return ResidualEstimate(np.full((2, 3), np.nan), status, message)


def contributions(labels: IntArray) -> FloatArray:
    """Each GROUP has denominator 2; overall is the mean of the two groups."""
    return np.stack(
        [
            np.stack(
                [(labels[:, 2 * g : 2 * g + 2] == c).sum(axis=1) / 2 for c in range(3)], axis=1
            )
            for g in range(2)
        ],
        axis=1,
    )


def srs(oracle: Oracle, k: int, seed: int) -> AuditDesign:
    n = len(oracle.labels)
    if n == 0 or k < 0 or k > n:
        raise ValueError("Require N > 0 and 0 <= k <= N")
    idx = np.sort(np.random.default_rng(seed).permutation(n)[:k])
    return AuditDesign(oracle.release_audit(idx), np.full(n, k / n, dtype=float))


def validate(observed: Observed, design: AuditDesign) -> None:
    n = len(observed.labels)
    audit = design.audit
    if n == 0 or design.pi.shape != (n,):
        raise ValueError("Empty population or invalid inclusion probabilities")
    if not np.isfinite(design.pi).all() or np.any((design.pi < 0) | (design.pi > 1)):
        raise ValueError("Inclusion probabilities must be in [0,1]")
    if audit.labels.shape != (len(audit.indices), 4):
        raise ValueError("Invalid audit shape")
    if not np.isin(audit.labels, [0, 1, 2]).all():
        raise ValueError("Invalid audit classes")
    if len(set(audit.indices)) != len(audit.indices):
        raise ValueError("Duplicate audit index")
    if np.any((audit.indices < 0) | (audit.indices >= n)):
        raise ValueError("Audit index out of range")
    if np.any(design.pi[audit.indices] <= 0):
        raise ValueError("Selected unit cannot have zero probability")
    exposed = observed.labels[audit.indices]
    if np.any((exposed >= 0) & (exposed != audit.labels)):
        raise ValueError("Audit contradicts visible labels")
    if not np.array_equal(observed.retained, (observed.labels >= 0).any(axis=1)):
        raise ValueError("Inconsistent S and visible labels")


def residual_estimate(
    observed: Observed, design: AuditDesign, prediction: FloatArray | None = None
) -> ResidualEstimate:
    validate(observed, design)
    hidden = (observed.labels < 0).any(axis=1)
    if np.any(hidden & (design.pi <= 0)):
        return unavailable("Unknown contributions have zero audit inclusion probability")
    visible = contributions(observed.labels)
    m = np.zeros_like(visible) if prediction is None else np.asarray(prediction)
    if m.shape != visible.shape or not np.isfinite(m).all():
        raise ValueError("Invalid missing-contribution prediction shape/values")
    if prediction is not None:
        missing_mass = (observed.labels.reshape(-1, 2, 2) < 0).sum(axis=2) / 2
        if np.any(m < -1e-12) or not np.allclose(m.sum(axis=2), missing_mass, atol=1e-10):
            raise ValueError("Prediction must describe only the missing probability mass")
    estimate = (visible + m).mean(axis=0)
    idx = design.audit.indices
    if len(idx):
        residual = contributions(design.audit.labels) - visible[idx] - m[idx]
        estimate += (residual / design.pi[idx, None, None]).sum(axis=0) / len(visible)
    return ResidualEstimate(estimate)


def post_stratified_m2(observed: Observed, design: AuditDesign) -> ResidualEstimate:
    validate(observed, design)
    if observed.mechanism != "M2":
        return unavailable("Post-stratification baseline is defined only for M2", "not_applicable")
    if not np.isin((observed.labels >= 0).sum(axis=1), [0, 4]).all():
        raise ValueError("M2 must be completely observed or hidden")
    n = len(observed.labels)
    deleted = ~observed.retained
    if not deleted.any():
        return ResidualEstimate(contributions(observed.labels).mean(axis=0))
    # This ratio is a SRS-within-layer post-stratified estimate, not a general unequal-pi HT.
    if not np.allclose(design.pi, design.pi[0]):
        raise ValueError("M2 post-stratified baseline currently requires global equal-pi SRS")
    pick = deleted[design.audit.indices]
    if not pick.any():
        return unavailable("Deleted stratum has no audited comparisons")
    known = contributions(observed.labels).sum(axis=0) / n
    missing = contributions(design.audit.labels[pick]).mean(axis=0) * deleted.mean()
    return ResidualEstimate(known + missing, diagnostics={"deleted_audits": int(pick.sum())})
