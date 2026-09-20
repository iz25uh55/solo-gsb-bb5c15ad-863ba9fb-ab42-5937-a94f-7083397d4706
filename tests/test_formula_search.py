"""Tests for the molecular-formula reverse search (branch and bound)."""

import pytest

from app.envelope import isotope_envelope, top_peaks
from app.formula_search import (
    MAX_VISITED_NODES,
    ElementSpec,
    SearchLimitExceeded,
    enumerate_compositions,
    formula_string,
    search_formulas,
)
from app.isotopes import most_abundant_isotope_mass
from app.parser import parse_formula

TOL = 0.0001
PRUNE = 1e-12


def _natural_peaks(formula: str, top: int = 10):
    env = isotope_envelope(parse_formula(formula), TOL, PRUNE)
    return top_peaks(env, top)


# --- formula_string ---------------------------------------------------------

def test_formula_string_lexicographic_symbols():
    assert formula_string({"C": 6, "H": 12, "O": 6}) == "C6H12O6"
    # Pure lexicographic order: "C" is a prefix of "Cl".
    assert formula_string({"C": 1, "Cl": 2, "H": 3}) == "CCl2H3"
    assert formula_string({"O": 1}) == "O"
    assert formula_string({"H": 0, "O": 1}) == "O"
    assert formula_string({}) == ""


# --- enumeration ------------------------------------------------------------

def test_enumerate_glucose_is_the_only_mass_hit():
    raw, visited, pruned = enumerate_compositions(
        180.063388,
        0.005,
        [
            ElementSpec("C", 0, 20),
            ElementSpec("H", 0, 40),
            ElementSpec("O", 0, 10),
        ],
    )
    assert len(raw) == 1
    counts, mass = raw[0]
    # Visit order is heaviest monoisotopic mass first: O, C, H.
    assert dict(zip(("O", "C", "H"), counts)) == {"O": 6, "C": 6, "H": 12}
    assert mass == pytest.approx(180.063388, abs=1e-5)
    assert visited >= 1
    assert pruned > 0


def test_enumerate_matches_brute_force():
    specs = [
        ElementSpec("C", 0, 4),
        ElementSpec("H", 0, 6),
        ElementSpec("N", 0, 3),
    ]
    target, tolerance = 42.0, 3.0
    raw, _, _ = enumerate_compositions(target, tolerance, specs)
    order = sorted(
        specs, key=lambda e: (-most_abundant_isotope_mass(e.symbol), e.symbol)
    )
    got = {frozenset(zip([e.symbol for e in order], c)) for c, _ in raw}

    expected = set()
    for c_c in range(5):
        for c_h in range(7):
            for c_n in range(4):
                mass = (
                    c_c * most_abundant_isotope_mass("C")
                    + c_h * most_abundant_isotope_mass("H")
                    + c_n * most_abundant_isotope_mass("N")
                )
                if mass > 0 and abs(mass - target) <= tolerance:
                    expected.add(frozenset(zip("CHN", (c_c, c_h, c_n))))
    assert got == expected


def test_enumerate_pruned_counter_accounts_for_skipped_counts():
    # Only count 1 is feasible for a single O at mass ~16: 5 of the 6
    # candidate count values are pruned; the root plus the one leaf are the
    # visited nodes.
    raw, visited, pruned = enumerate_compositions(
        most_abundant_isotope_mass("O"), 0.5, [ElementSpec("O", 0, 5)]
    )
    assert len(raw) == 1
    assert visited == 2
    assert pruned == 5


def test_enumerate_unreachable_window_prunes_everything_at_root():
    raw, visited, pruned = enumerate_compositions(
        500.0, 0.01, [ElementSpec("C", 0, 5), ElementSpec("H", 0, 5)]
    )
    assert raw == []
    assert visited == 1
    assert pruned > 0


def test_enumerate_respects_min_and_max_counts():
    # H2O would hit 18.01, but min_count H=3 forbids it.
    raw, _, _ = enumerate_compositions(
        18.01, 0.01, [ElementSpec("H", 3, 5), ElementSpec("O", 1, 1)]
    )
    assert raw == []

    # min_count forces atoms into every hit.
    raw, _, _ = enumerate_compositions(
        50.0, 5.0, [ElementSpec("N", 2, 4), ElementSpec("C", 0, 4)]
    )
    assert raw
    assert all(c[0] >= 2 for c, _ in raw)  # visit order: N first
    assert all(c[0] <= 4 and c[1] <= 4 for c, _ in raw)


def test_enumerate_excludes_all_zero_composition_under_huge_tolerance():
    raw, _, _ = enumerate_compositions(
        0.0, 1000.0, [ElementSpec("C", 0, 2), ElementSpec("H", 0, 2)]
    )
    assert raw
    assert all(any(c > 0 for c in counts) for counts, _ in raw)


def test_enumerate_zero_tolerance_needs_exact_mass():
    m_water = 2 * most_abundant_isotope_mass("H") + most_abundant_isotope_mass("O")
    raw, _, _ = enumerate_compositions(
        m_water, 0.0, [ElementSpec("H", 0, 5), ElementSpec("O", 0, 5)]
    )
    assert len(raw) == 1

    raw, _, _ = enumerate_compositions(
        m_water + 0.001, 0.0, [ElementSpec("H", 0, 5), ElementSpec("O", 0, 5)]
    )
    assert raw == []


def test_enumerate_tolerance_window_boundaries_are_inclusive():
    m_water = 2 * most_abundant_isotope_mass("H") + most_abundant_isotope_mass("O")
    for target in (m_water + 0.01, m_water - 0.01):
        raw, _, _ = enumerate_compositions(
            target, 0.01, [ElementSpec("H", 0, 5), ElementSpec("O", 0, 5)]
        )
        assert any(abs(m - m_water) < 1e-9 for _, m in raw)


def test_enumerate_independent_of_request_element_order():
    specs_a = [ElementSpec("N", 0, 3), ElementSpec("C", 0, 4), ElementSpec("H", 0, 6)]
    specs_b = list(reversed(specs_a))
    raw_a, va, pa = enumerate_compositions(42.0, 3.0, specs_a)
    raw_b, vb, pb = enumerate_compositions(42.0, 3.0, specs_b)

    def normalise(raw, specs):
        order = sorted(
            specs, key=lambda e: (-most_abundant_isotope_mass(e.symbol), e.symbol)
        )
        return {
            frozenset(zip([e.symbol for e in order], counts)): round(mass, 9)
            for counts, mass in raw
        }

    assert normalise(raw_a, specs_a) == normalise(raw_b, specs_b)
    assert va == vb and pa == pb


def test_enumerate_aborts_after_one_million_visited_nodes():
    # Windows wide open on six 0..200 elements: far beyond 1e6 nodes.
    with pytest.raises(SearchLimitExceeded) as excinfo:
        enumerate_compositions(
            60000.0,
            60000.0,
            [ElementSpec(symbol, 0, 200) for symbol in "CHONSP"],
        )
    assert excinfo.value.visited_nodes == MAX_VISITED_NODES + 1


# --- scoring/ranking --------------------------------------------------------

def test_search_scores_glucose_best_from_its_natural_envelope():
    observed = _natural_peaks("C6H12O6", top=8)
    result = search_formulas(
        180.063388,
        observed,
        [
            ElementSpec("C", 0, 12),
            ElementSpec("H", 0, 30),
            ElementSpec("N", 0, 3),
            ElementSpec("O", 0, 8),
        ],
        mass_tolerance=0.01,
        max_results=20,
        merge_tolerance=TOL,
        prune=PRUNE,
    )
    assert result.hits
    winner = result.hits[0]
    assert winner.formula == "C6H12O6"
    assert winner.composition == {"C": 6, "H": 12, "O": 6}
    # The observed list carries only the top 8 predicted peaks; every other
    # predicted peak is unmatched and counts at zero intensity (same MSE
    # convention as the enrichment/identify endpoints), so the winner is not
    # exactly zero but must be far better than any near-isobaric formula.
    assert winner.mass_error == pytest.approx(0.0, abs=1e-5)
    assert winner.mse == pytest.approx(0.0, abs=1e-3)
    assert result.hits[1].mse > winner.mse * 100
    # Ranking key: (mse, |mass error|, formula).
    keys = [(h.mse, abs(h.mass_error), h.formula) for h in result.hits]
    assert keys == sorted(keys)


def test_search_caps_returned_hits_but_reports_total():
    result = search_formulas(
        42.0,
        [(42.0, 100.0)],
        [ElementSpec("C", 0, 4), ElementSpec("H", 0, 6), ElementSpec("N", 0, 3)],
        mass_tolerance=3.0,
        max_results=2,
        merge_tolerance=TOL,
        prune=PRUNE,
    )
    assert result.total_hits == 18
    assert len(result.hits) == 2


def test_search_empty_hit_list_when_no_composition_in_window():
    result = search_formulas(
        500.0,
        [(500.0, 100.0)],
        [ElementSpec("C", 0, 5), ElementSpec("H", 0, 5)],
        mass_tolerance=0.01,
        max_results=10,
        merge_tolerance=TOL,
        prune=PRUNE,
    )
    assert result.hits == []
    assert result.total_hits == 0
    assert result.visited_nodes == 1
    assert result.pruned_branches > 0


def test_search_hit_pairs_one_to_one_with_observed_peaks():
    observed = _natural_peaks("C6H12O6", top=5)
    result = search_formulas(
        180.063388,
        observed,
        [ElementSpec("C", 4, 8), ElementSpec("H", 8, 16), ElementSpec("O", 4, 8)],
        mass_tolerance=0.01,
        max_results=5,
        merge_tolerance=TOL,
        prune=PRUNE,
    )
    hit = result.hits[0]
    # Every observed and every predicted peak participates in at least one pair.
    obs_indices = {i for i, _ in hit.pairs if i is not None}
    pred_indices = {j for _, j in hit.pairs if j is not None}
    assert obs_indices == set(range(len(observed)))
    assert len(pred_indices) <= len(hit.predicted)
    assert len(hit.pairs) >= 1
