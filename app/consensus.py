"""Consensus spectrum of replicate centroid measurements.

Given 2-20 centroid peak lists (replicate measurements of the same
sample), this module builds a consensus spectrum. Every spectrum is first
normalised to its own base peak (100); then all peaks of all spectra are
pooled onto one m/z axis and clustered by a deterministic complete-linkage
agglomerative clustering:

* a merge of two clusters is legal only when the merged cluster would not
  hold two peaks of the same input spectrum, and when its endpoint span
  (highest minus lowest member m/z) does not exceed
  ``max(tolerance_da, endpoint_mean * tolerance_ppm / 1e6)``, the endpoint
  mean being the average of the merged cluster's lowest and highest m/z;
* every round merges the best legal pair, compared lexicographically by
  (merged span, merged lowest m/z, sorted member (spectrum, peak)
  indices), and repeats until no legal merge remains.

Clusters covering at least ``min_replicates`` spectra are reported with

* the consensus m/z: the member m/z average weighted by the *original*
  (request) intensities — the arithmetic mean when every member intensity
  is zero;
* the member (spectrum, peak) indices;
* the median of the base-peak-normalised member intensities, counting
  every spectrum absent from the cluster as 0 (the median runs over all
  input spectra),

sorted by ascending consensus m/z.
"""

from __future__ import annotations

import heapq
from bisect import bisect_left, insort
from dataclasses import dataclass
from statistics import median

# Accepted request bounds.
MIN_SPECTRA = 2
MAX_SPECTRA = 20
MIN_PEAKS = 1
MAX_PEAKS = 200
MIN_REPLICATES = 2


class ConsensusError(ValueError):
    """Validation error for the consensus-spectrum request.

    Carries a machine-readable ``error`` code plus the 0-based
    ``spectrum_index``/``peak_index`` locating the offending peak
    (``None`` when not index-specific).
    """

    def __init__(
        self,
        error: str,
        message: str,
        spectrum_index: int | None = None,
        peak_index: int | None = None,
    ) -> None:
        super().__init__(message)
        self.error = error
        self.message = message
        self.spectrum_index = spectrum_index
        self.peak_index = peak_index


@dataclass(frozen=True, order=True)
class ConsensusMember:
    """One member peak of a consensus cluster."""

    spectrum_index: int
    peak_index: int


@dataclass(frozen=True)
class ConsensusCluster:
    mz: float
    intensity: float
    members: tuple[ConsensusMember, ...]


@dataclass(frozen=True)
class ConsensusResult:
    spectrum_count: int
    tolerance_da: float
    tolerance_ppm: float
    min_replicates: int
    peak_count: int
    clusters: tuple[ConsensusCluster, ...]


def _validate_spectra(spectra: list[list[tuple[float, float]]]) -> None:
    """Require strictly increasing m/z per spectrum and a positive base peak."""
    for s_idx, peaks in enumerate(spectra):
        for p_idx in range(1, len(peaks)):
            prev_mz, cur_mz = peaks[p_idx - 1][0], peaks[p_idx][0]
            if cur_mz == prev_mz:
                raise ConsensusError(
                    "invalid_peaks",
                    f"Duplicate peak m/z {cur_mz} in spectrum {s_idx}; "
                    f"m/z values must be strictly increasing",
                    s_idx,
                    p_idx,
                )
            if cur_mz < prev_mz:
                raise ConsensusError(
                    "invalid_peaks",
                    f"Peak m/z values of spectrum {s_idx} must be sorted in "
                    f"increasing order; m/z {cur_mz} at index {p_idx} "
                    f"follows {prev_mz}",
                    s_idx,
                    p_idx,
                )
        if max(intensity for _, intensity in peaks) <= 0.0:
            raise ConsensusError(
                "zero_spectrum",
                f"Spectrum {s_idx} has no positive-intensity peak; the "
                f"base-peak normalisation requires at least one",
                s_idx,
            )


def build_consensus_spectrum(
    spectra: list[list[tuple[float, float]]],
    tolerance_da: float,
    tolerance_ppm: float,
    min_replicates: int,
) -> ConsensusResult:
    """Cluster pooled centroid peaks into a replicate consensus spectrum.

    ``spectra`` holds 2-20 peak lists of ``(mz, intensity)`` tuples, each
    strictly m/z-increasing with non-negative intensities and at least one
    positive intensity (validated here). See the module docstring for the
    clustering and reporting rules.
    """
    _validate_spectra(spectra)
    spectrum_count = len(spectra)
    normalised: list[list[float]] = []
    for peaks in spectra:
        base = max(intensity for _, intensity in peaks)
        normalised.append([100.0 * intensity / base for _, intensity in peaks])

    # Agglomerative cluster state, indexed by cluster id. Singletons are
    # created in global m/z order; merged clusters are appended.
    members: list[tuple[tuple[int, int], ...]] = []
    spectra_of: list[frozenset[int]] = []
    lo: list[float] = []
    hi: list[float] = []
    alive: list[bool] = []
    # (lo, cluster id) of every cluster ever created, kept sorted; dead
    # clusters stay in the list and are skipped when scanning.
    ordered: list[tuple[float, int]] = []

    singletons = [
        (mz, s_idx, p_idx)
        for s_idx, peaks in enumerate(spectra)
        for p_idx, (mz, _) in enumerate(peaks)
    ]
    # Sort by m/z, then spectrum and peak index for a deterministic order.
    singletons.sort()
    for mz, s_idx, p_idx in singletons:
        members.append(((s_idx, p_idx),))
        spectra_of.append(frozenset((s_idx,)))
        lo.append(mz)
        hi.append(mz)
        alive.append(True)
        ordered.append((mz, len(members) - 1))

    # Any legal pair's span is bounded by ``window``: a legal span is at
    # most max(tolerance_da, endpoint_mean * ppm/1e6) and the endpoint mean
    # never exceeds the global maximum m/z.
    max_mz = singletons[-1][0] if singletons else 0.0
    window = max(tolerance_da, max_mz * tolerance_ppm / 1e6)

    def evaluate(a: int, b: int):
        """Heap key for merging clusters ``a``/``b``, or None when illegal."""
        if spectra_of[a] & spectra_of[b]:
            return None
        merged_lo = lo[a] if lo[a] < lo[b] else lo[b]
        merged_hi = hi[a] if hi[a] > hi[b] else hi[b]
        span = merged_hi - merged_lo
        limit = tolerance_da
        ppm_limit = (merged_lo + merged_hi) / 2.0 * tolerance_ppm / 1e6
        if ppm_limit > limit:
            limit = ppm_limit
        if span > limit:
            return None
        merged_members = tuple(sorted(members[a] + members[b]))
        first, second = (a, b) if a < b else (b, a)
        return (span, merged_lo, merged_members, first, second)

    def pairs_for(z: int, newer_only: bool) -> list:
        """Legal merge entries between cluster ``z`` and its neighbours.

        Only clusters whose lowest m/z falls inside
        ``[lo[z] - window, hi[z] + window]`` can satisfy the span limit, so
        the scan restricts to that slice of ``ordered``.
        """
        entries = []
        left = bisect_left(ordered, (lo[z] - window, -1))
        right = bisect_left(ordered, (hi[z] + window, len(members)))
        for k in range(left, right):
            _, cid = ordered[k]
            if cid == z or not alive[cid] or (newer_only and cid < z):
                continue
            entry = evaluate(z, cid)
            if entry is not None:
                entries.append(entry)
        return entries

    # Seed the heap with every legal singleton pair (each unordered pair
    # exactly once); merged clusters later push their own pairs.
    heap: list = []
    for z in range(len(members)):
        heap.extend(pairs_for(z, newer_only=True))
    heapq.heapify(heap)

    while heap:
        _, _, merged_members, a, b = heapq.heappop(heap)
        if not (alive[a] and alive[b]):
            # Stale entry: one side already merged into another cluster.
            continue
        alive[a] = False
        alive[b] = False
        cid = len(members)
        members.append(merged_members)
        spectra_of.append(spectra_of[a] | spectra_of[b])
        merged_lo = lo[a] if lo[a] < lo[b] else lo[b]
        merged_hi = hi[a] if hi[a] > hi[b] else hi[b]
        lo.append(merged_lo)
        hi.append(merged_hi)
        alive.append(True)
        insort(ordered, (merged_lo, cid))
        # The new cluster is the highest id, so none of its pairs can have
        # been pushed before.
        for entry in pairs_for(cid, newer_only=False):
            heapq.heappush(heap, entry)

    clusters: list[ConsensusCluster] = []
    for cid in range(len(members)):
        if not alive[cid]:
            continue
        cluster_members = members[cid]
        # One peak per spectrum per cluster, so the member count is the
        # number of covered spectra.
        if len(cluster_members) < min_replicates:
            continue
        weights = [spectra[s][p][1] for s, p in cluster_members]
        total_weight = sum(weights)
        if total_weight > 0.0:
            mz = (
                sum(
                    spectra[s][p][0] * weight
                    for (s, p), weight in zip(cluster_members, weights)
                )
                / total_weight
            )
        else:
            # All member intensities are zero: fall back to the plain mean.
            mz = sum(spectra[s][p][0] for s, p in cluster_members) / len(
                cluster_members
            )
        values = [0.0] * spectrum_count
        for s, p in cluster_members:
            values[s] = normalised[s][p]
        clusters.append(
            ConsensusCluster(
                mz=mz,
                intensity=median(values),
                members=tuple(
                    ConsensusMember(spectrum_index=s, peak_index=p)
                    for s, p in cluster_members
                ),
            )
        )
    clusters.sort(key=lambda cluster: (cluster.mz, cluster.members))

    return ConsensusResult(
        spectrum_count=spectrum_count,
        tolerance_da=tolerance_da,
        tolerance_ppm=tolerance_ppm,
        min_replicates=min_replicates,
        peak_count=sum(len(peaks) for peaks in spectra),
        clusters=tuple(clusters),
    )
