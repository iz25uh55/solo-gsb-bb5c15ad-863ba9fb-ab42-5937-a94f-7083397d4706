"""Replicate consensus spectrum from repeated centroid measurements.

Each input replicate is base-peak normalised on its own (base peak = 100).
All peaks are then pooled, sorted by m/z (spectrum and peak index break
m/z ties), and fused with a deterministic *complete-linkage* agglomerative
clustering. Two current clusters may merge only when:

* the merged cluster would not contain two peaks from the same input
  spectrum (each spectrum contributes at most one member per cluster); and
* the span between the extreme member m/z values of the merged cluster
  does not exceed
  ``max(tolerance_da, endpoint_mean * tolerance_ppm / 1e6)``.

Every round the globally best legal merge is performed. Candidate merges
rank by (merged span, lowest member m/z, sorted member-spectrum tuple,
sorted (spectrum, peak-index) member tuple); the member tuple alone is a
strict order (two distinct cluster pairs cannot have the same merged
member set), so cluster ids only complete the heap key. The process stops
once no legal merge remains.

The implementation is the classic lazy best-merge agglomerative scheme:
legal cluster pairs sit on a key-ordered heap and entries whose clusters
have since been merged are discarded on pop; after every merge the new
cluster is paired with every *windowed* live cluster (peaks within one
spectrum can never merge, and the span only ever grows, so both checks
prune permanently). Spectrum membership is a bitmask (at most 20 spectra)
and live clusters are kept sorted by lowest member m/z so the window is a
pair of binary searches.

Clusters covering at least ``min_replicates`` distinct spectra survive.
Each consensus peak reports the original-intensity-weighted mean m/z, the
member indices and the median of the per-spectrum base-peak-normalised
intensities over *all* input spectra (spectra without a member contribute
zero), sorted by m/z.
"""

from __future__ import annotations

import heapq
from bisect import bisect_left, bisect_right, insort
from dataclasses import dataclass
from statistics import median

# Upper bound on peaks per input spectrum (mirrored by the request model).
MAX_PEAKS_PER_SPECTRUM = 200
# Upper bound on the number of input spectra (mirrored by the model).
MAX_SPECTRA = 20
# Safety stop on the number of legal merge candidates enumerated (an
# absurdly large tolerance would otherwise pair almost every peak).
MAX_MERGE_CANDIDATES = 2_000_000


class ConsensusError(ValueError):
    """Validation error for a consensus-spectrum request.

    Carries a machine-readable ``error`` code, the offending request
    ``field``, a human-readable ``message`` and, when applicable, the
    0-based ``spectrum`` index and 0-based peak ``index`` inside it.
    """

    def __init__(
        self,
        error: str,
        field_: str,
        message: str,
        *,
        spectrum: int | None = None,
        index: int | None = None,
    ) -> None:
        super().__init__(message)
        self.error = error
        self.field = field_
        self.message = message
        self.spectrum = spectrum
        self.index = index


@dataclass(frozen=True)
class ConsensusMember:
    """One consensus-peak member: a peak in one input spectrum."""

    spectrum: int
    peak_index: int


@dataclass(frozen=True)
class ConsensusPeak:
    mz: float
    replicate_count: int
    members: tuple[ConsensusMember, ...]
    median_intensity: float


@dataclass(frozen=True)
class ConsensusResult:
    spectrum_count: int
    total_peak_count: int
    tolerance_da: float
    tolerance_ppm: float
    min_replicates: int
    peaks: tuple[ConsensusPeak, ...]


def build_consensus_spectrum(
    spectra: list[list[tuple[float, float]]],
    tolerance_da: float,
    tolerance_ppm: float,
    min_replicates: int,
) -> ConsensusResult:
    """Build the replicate consensus spectrum.

    ``spectra`` are lists of strictly m/z-increasing ``(mz, intensity)``
    peaks with non-negative intensities; every spectrum must contain at
    least one positive-intensity peak. Tolerances and ``min_replicates``
    are validated by the endpoint.
    """
    n = len(spectra)

    # Base-peak normalise each spectrum independently (base peak = 100).
    normed: list[list[float]] = []
    for s_idx, peaks in enumerate(spectra):
        base = max(intensity for _, intensity in peaks)
        if base <= 0.0:
            raise ConsensusError(
                "zero_spectrum",
                "spectra",
                f"Spectrum {s_idx} has no positive-intensity peak; each "
                "spectrum needs a base peak to normalise against",
                spectrum=s_idx,
            )
        normed.append([100.0 * intensity / base for _, intensity in peaks])

    # Pool every peak and sort by m/z (spectrum and peak index break ties).
    pooled: list[tuple[float, int, int, float]] = []
    for s_idx, peaks in enumerate(spectra):
        for p_idx, (mz, intensity) in enumerate(peaks):
            pooled.append((mz, s_idx, p_idx, intensity))
    pooled.sort(key=lambda item: (item[0], item[1], item[2]))
    total = len(pooled)

    # Flat cluster tables; cluster id is the table index. Singletons are
    # ids 0..total-1 in pooled order; every merge appends a new id.
    c_members: list[tuple[tuple[int, int], ...]] = [()] * total
    c_mask: list[int] = [0] * total
    c_lo: list[float] = [0.0] * total
    c_hi: list[float] = [0.0] * total
    c_weighted: list[float] = [0.0] * total
    c_weight: list[float] = [0.0] * total
    c_alive: list[bool] = [True] * total
    pooled_mz: list[float] = [0.0] * total
    for cid, (mz, s_idx, p_idx, intensity) in enumerate(pooled):
        c_members[cid] = ((s_idx, p_idx),)
        c_mask[cid] = 1 << s_idx
        c_lo[cid] = c_hi[cid] = mz
        c_weighted[cid] = mz * intensity
        c_weight[cid] = intensity
        pooled_mz[cid] = mz

    ppm_fraction = tolerance_ppm * 1e-6
    budget = [MAX_MERGE_CANDIDATES]
    heap: list[tuple[tuple, int, int]] = []

    def note_candidate() -> None:
        budget[0] -= 1
        if budget[0] < 0:
            raise ConsensusError(
                "too_many_candidates",
                "tolerance_da",
                f"Consensus clustering enumerated more than "
                f"{MAX_MERGE_CANDIDATES:,} legal merge candidates; reduce "
                "tolerance_da or tolerance_ppm",
            )

    def span_of(a: int, b: int) -> float | None:
        """Merged endpoint span if the pair satisfies both hard rules."""
        # Complete-linkage rule 1: at most one peak per input spectrum.
        if c_mask[a] & c_mask[b]:
            return None
        lo = c_lo[a] if c_lo[a] < c_lo[b] else c_lo[b]
        hi = c_hi[a] if c_hi[a] > c_hi[b] else c_hi[b]
        span = hi - lo
        # Complete-linkage rule 2: endpoints within the max-combined
        # absolute / relative tolerance.
        if span > max(tolerance_da, (lo + hi) * 0.5 * ppm_fraction):
            return None
        return span

    def _merge_members(a: int, b: int) -> tuple:
        """Linear merge of the two disjoint, already-sorted member tuples."""
        left = c_members[a]
        right = c_members[b]
        merged: list[tuple[int, int]] = []
        ia = ib = 0
        while ia < len(left) and ib < len(right):
            if left[ia] < right[ib]:
                merged.append(left[ia])
                ia += 1
            else:
                merged.append(right[ib])
                ib += 1
        merged.extend(left[ia:])
        merged.extend(right[ib:])
        return tuple(merged)

    def push_pair(a: int, b: int) -> None:
        span = span_of(a, b)
        if span is None:
            return
        note_candidate()
        lo = c_lo[a] if c_lo[a] < c_lo[b] else c_lo[b]
        # The member tuple is a strict order on distinct cluster pairs and
        # already implies the member-spectrum tuple, so it alone settles
        # every (span, lo) tie; the ids only complete the heap key.
        key = (span, lo, _merge_members(a, b), min(a, b), max(a, b))
        heapq.heappush(heap, (key, a, b))

    def window_bounds(lo: float, hi: float) -> tuple[float, float]:
        """Conservative m/z window a legal partner's own lowest m/z lies in.

        Given a live cluster spanning ``[lo, hi]`` and a partner with
        lowest m/z ``p``, the exact complete-linkage span rule against the
        merged endpoints gives these *necessary* bounds (any legal partner
        satisfies them, so the window never drops one; the exact tolerance
        is rechecked per pair):

        * right partner (merged extents ``[lo, p']``): the Da branch needs
          ``p' <= lo + tol`` and the ppm branch
          ``p' <= lo(2+p)/(2-p)``;
        * left partner (merged extents ``[p, hi]``): the Da branch needs
          ``p >= lo - tol`` and the ppm branch
          ``p >= (2 lo - p·hi)/(2 + p)``.

        The window is keyed on each live cluster's own lowest m/z, which is
        the stricter (inward) endpoint for a right-side partner and the
        exact endpoint for a left-side one. With a relative fraction
        at/above 1 the ppm window is unbounded.
        """
        if ppm_fraction >= 1.0:
            return float("-inf"), float("inf")
        right = max(
            lo + tolerance_da,
            lo * (2.0 + ppm_fraction) / (2.0 - ppm_fraction),
        )
        left = min(
            lo - tolerance_da,
            (2.0 * lo - ppm_fraction * hi) / (2.0 + ppm_fraction),
        )
        return left, right

    # Seed the heap with legal singleton pairs inside the conservative
    # m/z window. The span only grows under merging, so a pair outside the
    # window at the start stays illegal; span_of rechecks the tolerance.
    for i in range(total):
        mz_i = pooled_mz[i]
        _lo_bound, hi_bound = window_bounds(mz_i, mz_i)
        j_hi = bisect_right(pooled_mz, hi_bound, lo=i + 1)
        for j in range(i + 1, j_hi):
            push_pair(i, j)

    # Live clusters kept sorted by (lowest member m/z, id) for windowed
    # partner scans; starts as the singletons in pooled order.
    live: list[tuple[float, int]] = [(pooled_mz[i], i) for i in range(total)]

    # Deterministic complete-linkage rounds: best legal merge first.
    while heap:
        _, a, b = heapq.heappop(heap)
        if not c_alive[a] or not c_alive[b]:
            continue

        cid = len(c_mask)
        mask = c_mask[a] | c_mask[b]
        lo = c_lo[a] if c_lo[a] < c_lo[b] else c_lo[b]
        hi = c_hi[a] if c_hi[a] > c_hi[b] else c_hi[b]
        c_members.append(_merge_members(a, b))
        c_mask.append(mask)
        c_lo.append(lo)
        c_hi.append(hi)
        c_weighted.append(c_weighted[a] + c_weighted[b])
        c_weight.append(c_weight[a] + c_weight[b])
        c_alive.append(True)
        c_alive[a] = c_alive[b] = False

        # Drop the two merged clusters from the sorted live list.
        for dead in (a, b):
            pos = bisect_left(live, (c_lo[dead], dead))
            # The (lo, id) pair is unique; guard against float edge cases.
            while live[pos][1] != dead:
                pos += 1
            live.pop(pos)

        # Pair the new cluster with live clusters whose own lowest m/z
        # falls in the conservative window; a full-spectrum mask can never
        # merge with anything. The window catches left-side, right-side and
        # straddling partners (see window_bounds), and span_of rechecks the
        # exact tolerance.
        if mask != (1 << n) - 1:
            lo_bound, hi_bound = window_bounds(lo, hi)
            start = bisect_left(live, (lo_bound, -1))
            stop = bisect_right(live, (hi_bound, 1 << 30))
            for pos in range(start, stop):
                other = live[pos][1]
                push_pair(cid, other)

        # Keep the new cluster even when it cannot merge further: it is a
        # final cluster and must be reported/filtered like any other.
        insort(live, (lo, cid))

    # Keep clusters with enough replicate coverage and build the output.
    consensus_peaks: list[ConsensusPeak] = []
    for cid in range(len(c_mask)):
        if not c_alive[cid]:
            continue
        replicate_count = c_mask[cid].bit_count()
        if replicate_count < min_replicates:
            continue

        norm_by_spectrum = {
            s_idx: normed[s_idx][p_idx] for s_idx, p_idx in c_members[cid]
        }
        # Missing spectra explicitly count at zero intensity.
        intensities = [norm_by_spectrum.get(s_idx, 0.0) for s_idx in range(n)]
        median_intensity = median(intensities)

        if c_weight[cid] > 0.0:
            mz = c_weighted[cid] / c_weight[cid]
        else:
            # Every member happens to be a zero-intensity peak: fall back
            # to the unweighted mean member m/z.
            mz = sum(spectra[s][p][0] for s, p in c_members[cid]) / len(
                c_members[cid]
            )

        members = tuple(
            ConsensusMember(spectrum=s_idx, peak_index=p_idx)
            for s_idx, p_idx in c_members[cid]
        )
        consensus_peaks.append(
            ConsensusPeak(
                mz=mz,
                replicate_count=replicate_count,
                members=members,
                median_intensity=median_intensity,
            )
        )

    consensus_peaks.sort(
        key=lambda peak: (
            peak.mz,
            tuple((m.spectrum, m.peak_index) for m in peak.members),
        )
    )

    return ConsensusResult(
        spectrum_count=n,
        total_peak_count=total,
        tolerance_da=tolerance_da,
        tolerance_ppm=tolerance_ppm,
        min_replicates=min_replicates,
        peaks=tuple(consensus_peaks),
    )
