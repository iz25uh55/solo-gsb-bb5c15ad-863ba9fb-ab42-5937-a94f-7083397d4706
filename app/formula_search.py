"""Molecular-formula reverse search by branch-and-bound enumeration.

Given a target monoisotopic mass, observed isotope peaks and per-element
count ranges (2-6 distinct elements, each count between a ``min_count`` and
a ``max_count``), every composition whose monoisotopic mass falls within the
mass tolerance is enumerated depth-first. Elements are visited from the
heaviest monoisotopic mass down; at every node the minimum and maximum mass
still obtainable from the not-yet-assigned elements restricts the current
element's integer count window, so counts that cannot possibly land inside
the tolerance window are pruned without recursion.

Each hit reuses the existing natural-abundance isotope convolution,
base-peak normalisation and one-to-one peak matching (unmatched peaks
counted at zero intensity); candidates rank by MSE, then absolute mass
error, then the element-symbol-lexicographic formula string. Searches
visiting more than :data:`MAX_VISITED_NODES` nodes abort with
:class:`SearchLimitExceeded` so an overly wide request cannot exhaust the
worker.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .enrichment import normalise_base_peak, pair_and_score
from .envelope import Peak, isotope_envelope
from .isotopes import most_abundant_isotope_mass
from .parser import ParsedFormula

# Hard cap on enumeration nodes (the root counts as one). Exceeding it
# aborts the search and the endpoint answers HTTP 422.
MAX_VISITED_NODES = 1_000_000

# Absolute guard (Da) against ordinary floating-point round-off when
# deriving integer count windows and deciding window membership. It only
# ever keeps boundary-neighbour counts (which are re-checked at the leaf);
# at 1e-10 Da it is far smaller than any physically meaningful mass
# tolerance and cannot include an off-by-one-atom neighbour (the lightest
# element, H, shifts the mass by ~1 Da per count).
_ROUND_EPS = 1e-10


class SearchLimitExceeded(Exception):
    """Enumeration exceeded :data:`MAX_VISITED_NODES` visited nodes."""

    def __init__(self, visited_nodes: int, pruned_branches: int) -> None:
        super().__init__(
            f"Formula search visited {visited_nodes} nodes; the limit is "
            f"{MAX_VISITED_NODES}"
        )
        self.visited_nodes = visited_nodes
        self.pruned_branches = pruned_branches


@dataclass(frozen=True)
class ElementSpec:
    """One search element with its inclusive atom-count range."""

    symbol: str
    min_count: int
    max_count: int


@dataclass
class FormulaHit:
    """One within-tolerance composition scored against the observed peaks."""

    formula: str
    composition: dict[str, int]
    monoisotopic_mass: float
    mass_error: float  # predicted monoisotopic mass - target mass (Da)
    mse: float
    pairs: list[tuple[int | None, int | None]]
    predicted: list[Peak]  # base-peak normalised envelope


@dataclass
class FormulaSearchResult:
    """Ranked hits (capped at the requested ``max_results``) plus search stats."""

    hits: list[FormulaHit]
    total_hits: int
    visited_nodes: int
    pruned_branches: int


def formula_string(composition: dict[str, int]) -> str:
    """Canonical formula: element symbols in lexicographic order, counts of
    1 omitted and zero-count elements skipped."""
    return "".join(
        symbol + (str(count) if count > 1 else "")
        for symbol, count in sorted(composition.items())
        if count > 0
    )


def enumerate_compositions(
    target_mass: float,
    mass_tolerance: float,
    elements: list[ElementSpec],
) -> tuple[list[tuple[tuple[int, ...], float]], int, int]:
    """Branch-and-bound enumeration of compositions inside the mass window.

    Elements are visited heaviest-mononisotopic-mass first (ties by symbol)
    so the suffix mass bounds tighten as early as possible. Returns the raw
    hits as ``((counts in visit order, monoisotopic mass), ...)`` plus the
    visited-node and pruned-branch counters. Raises
    :class:`SearchLimitExceeded` past :data:`MAX_VISITED_NODES` visits.
    """
    specs = sorted(
        elements,
        key=lambda e: (-most_abundant_isotope_mass(e.symbol), e.symbol),
    )
    masses = [most_abundant_isotope_mass(s.symbol) for s in specs]
    n = len(specs)

    # suffix_min[i]/suffix_max[i]: smallest/largest mass the elements
    # i..n-1 can still contribute given their count ranges.
    suffix_min = [0.0] * (n + 1)
    suffix_max = [0.0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_min[i] = suffix_min[i + 1] + specs[i].min_count * masses[i]
        suffix_max[i] = suffix_max[i + 1] + specs[i].max_count * masses[i]

    visited_nodes = 0
    pruned_branches = 0
    raw_hits: list[tuple[tuple[int, ...], float]] = []

    def visit(index: int, counts: list[int], partial_mass: float) -> None:
        nonlocal visited_nodes, pruned_branches
        visited_nodes += 1
        if visited_nodes > MAX_VISITED_NODES:
            raise SearchLimitExceeded(visited_nodes, pruned_branches)

        if index == n:
            # Re-check membership with a pure round-off guard: count-window
            # widening at the parent may have reached boundary neighbours.
            if (
                partial_mass > 0.0
                and abs(partial_mass - target_mass) <= mass_tolerance + _ROUND_EPS
            ):
                raw_hits.append((tuple(counts), partial_mass))
            return

        mass_i = masses[index]
        # For count c of this element to leave a feasible completion:
        #   partial + c*m_i + suffix_max[i+1] >= target - tolerance
        #   partial + c*m_i + suffix_min[i+1] <= target + tolerance
        low = target_mass - mass_tolerance - partial_mass - suffix_max[index + 1]
        high = target_mass + mass_tolerance - partial_mass - suffix_min[index + 1]
        window_lo = math.ceil((low - _ROUND_EPS) / mass_i)
        window_hi = math.floor((high + _ROUND_EPS) / mass_i)

        lo = max(specs[index].min_count, window_lo)
        hi = min(specs[index].max_count, window_hi)
        candidate_count = specs[index].max_count - specs[index].min_count + 1
        visited_count = max(0, hi - lo + 1)
        # Every count value outside the feasible window is one pruned
        # branch (the empty-window case included, via this subtraction).
        pruned_branches += candidate_count - visited_count

        for count in range(lo, hi + 1):
            counts.append(count)
            visit(index + 1, counts, partial_mass + count * mass_i)
            counts.pop()

    visit(0, [], 0.0)
    return raw_hits, visited_nodes, pruned_branches


def _score_hit(
    counts: tuple[int, ...],
    mono_mass: float,
    target_mass: float,
    specs: list[ElementSpec],
    observed: list[Peak],
    mass_tolerance: float,
    merge_tolerance: float,
    prune: float,
) -> FormulaHit:
    composition = {
        specs[i].symbol: count for i, count in enumerate(counts) if count > 0
    }
    parsed = ParsedFormula(composition=composition, labels={})
    envelope = isotope_envelope(parsed, merge_tolerance, prune)
    predicted = normalise_base_peak(envelope)
    pairs, mse = pair_and_score(observed, predicted, mass_tolerance)
    return FormulaHit(
        formula=formula_string(composition),
        composition=composition,
        monoisotopic_mass=mono_mass,
        mass_error=mono_mass - target_mass,
        mse=mse,
        pairs=pairs,
        predicted=predicted,
    )


def search_formulas(
    target_mass: float,
    observed: list[Peak],
    elements: list[ElementSpec],
    mass_tolerance: float,
    max_results: int,
    merge_tolerance: float,
    prune: float,
) -> FormulaSearchResult:
    """Enumerate within-tolerance compositions and score them against the
    base-peak-normalised observed peaks. Hits are ranked by ascending MSE,
    then absolute mass error, then lexicographic formula, and capped at
    ``max_results``; the total hit count is returned separately."""
    obs = normalise_base_peak(observed)
    raw_hits, visited_nodes, pruned_branches = enumerate_compositions(
        target_mass, mass_tolerance, elements
    )

    specs = sorted(
        elements,
        key=lambda e: (-most_abundant_isotope_mass(e.symbol), e.symbol),
    )
    hits: list[FormulaHit] = []
    for counts, mono_mass in raw_hits:
        hit = _score_hit(
            counts,
            mono_mass,
            target_mass,
            specs,
            obs,
            mass_tolerance,
            merge_tolerance,
            prune,
        )
        hits.append(hit)

    hits.sort(key=lambda h: (h.mse, abs(h.mass_error), h.formula))
    return FormulaSearchResult(
        hits=hits[:max_results],
        total_hits=len(hits),
        visited_nodes=visited_nodes,
        pruned_branches=pruned_branches,
    )


__all__ = [
    "MAX_VISITED_NODES",
    "ElementSpec",
    "FormulaHit",
    "FormulaSearchResult",
    "SearchLimitExceeded",
    "enumerate_compositions",
    "formula_string",
    "search_formulas",
]
