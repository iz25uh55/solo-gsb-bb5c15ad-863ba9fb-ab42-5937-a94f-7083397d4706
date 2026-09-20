"""Tests for the replicate consensus-spectrum builder and its endpoint."""

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.consensus import (
    ConsensusError,
    build_consensus_spectrum,
)


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "db_path", str(tmp_path / "test.db"))
    from app.main import app

    with TestClient(app) as c:
        yield c


def _payload(spectra, tol_da=0.01, tol_ppm=5.0, min_replicates=2):
    return {
        "spectra": [
            {"peaks": [{"mz": mz, "intensity": i} for mz, i in peaks]}
            for peaks in spectra
        ],
        "tolerance_da": tol_da,
        "tolerance_ppm": tol_ppm,
        "min_replicates": min_replicates,
    }


# --- unit-level builder -----------------------------------------------------

def test_three_replicates_merge_and_weighted_mz():
    spectra = [
        [(100.0, 1000.0), (200.0, 500.0)],
        [(100.002, 500.0), (200.001, 1000.0)],
        [(100.004, 200.0), (110.2, 50.0), (200.0, 800.0)],
    ]
    result = build_consensus_spectrum(spectra, 0.01, 5.0, 2)
    assert len(result.peaks) == 2
    first, second = result.peaks
    assert first.replicate_count == 3
    # Original-intensity-weighted m/z.
    expected = (100.0 * 1000 + 100.002 * 500 + 100.004 * 200) / 1700
    assert first.mz == pytest.approx(expected, abs=1e-9)
    assert [(m.spectrum, m.peak_index) for m in first.members] == [
        (0, 0),
        (1, 0),
        (2, 0),
    ]
    # Normalised intensities at m/z 100: spectrum 0 base 1000 -> 100,
    # spectrum 1 base 1000 (at m/z 200) -> 50, spectrum 2 base 800 -> 25;
    # median([100, 50, 25]) = 50.
    assert first.median_intensity == pytest.approx(50.0)
    # At m/z 200: [50, 100, 100]; median 100.
    assert second.median_intensity == pytest.approx(100.0)
    # Sorted by m/z.
    assert result.peaks[0].mz < result.peaks[1].mz


def test_same_spectrum_two_peaks_never_one_cluster():
    spectra = [
        [(100.0, 100.0), (100.001, 50.0), (200.0, 20.0)],
        [(100.0005, 100.0), (200.0005, 40.0)],
    ]
    result = build_consensus_spectrum(spectra, 0.01, 5.0, 2)
    member_sets = [
        {(m.spectrum, m.peak_index) for m in p.members} for p in result.peaks
    ]
    # No cluster contains two members sharing a spectrum index.
    for peak in result.peaks:
        spectra_used = [m.spectrum for m in peak.members]
        assert len(spectra_used) == len(set(spectra_used))
    assert {(0, 0), (1, 0)} in member_sets
    assert {(0, 2), (1, 1)} in member_sets
    # The lone spectrum-0 peak at 100.001 forms no replicate cluster.
    assert all((0, 1) not in s for s in member_sets)


def test_min_replicates_threshold_filters_singletons():
    spectra = [
        [(100.0, 100.0), (150.0, 10.0)],
        [(100.001, 100.0)],
    ]
    assert len(build_consensus_spectrum(spectra, 0.01, 5.0, 2).peaks) == 1
    result = build_consensus_spectrum(spectra, 0.01, 5.0, 1)
    # The singleton peak at 150 survives at min_replicates = 1.
    assert len(result.peaks) == 2
    singleton = next(p for p in result.peaks if abs(p.mz - 150.0) < 1e-9)
    assert singleton.replicate_count == 1


def test_missing_spectra_count_as_zero_in_median():
    # Two of four spectra carry the peak; normalised values 100 and 100,
    # the two missing spectra count as 0 -> median([100, 0, 100, 0]) = 50.
    spectra = [
        [(100.0, 100.0), (300.0, 10.0)],
        [(250.0, 100.0)],
        [(100.001, 100.0), (260.0, 60.0)],
        [(251.0, 100.0)],
    ]
    result = build_consensus_spectrum(spectra, 0.01, 5.0, 2)
    near100 = next(p for p in result.peaks if abs(p.mz - 100.0) < 0.01)
    assert near100.replicate_count == 2
    assert near100.median_intensity == pytest.approx(50.0)


def test_base_peak_normalisation_is_per_spectrum():
    # The 100.001 peak is 10% of its spectrum's base but 100% of another.
    spectra = [
        [(100.0, 1000.0)],
        [(100.001, 100.0)],
    ]
    result = build_consensus_spectrum(spectra, 0.01, 5.0, 2)
    peak = result.peaks[0]
    # Median of [100, 100] (each spectrum normalised against its own base).
    assert peak.median_intensity == pytest.approx(100.0)
    # Weighted m/z uses the original raw intensities (1000 vs 100).
    assert peak.mz == pytest.approx(
        (100.0 * 1000 + 100.001 * 100) / 1100, abs=1e-12
    )


def test_ppm_tolerance_dominates_at_high_mz():
    spectra = [
        [(600.0, 100.0)],
        [(600.005, 100.0)],
    ]
    # 0.005 Da gap: 0.001 Da absolute blocks, 10 ppm (=0.006 Da) allows.
    assert len(build_consensus_spectrum(spectra, 0.001, 10.0, 2).peaks) == 1
    assert len(build_consensus_spectrum(spectra, 0.001, 1.0, 2).peaks) == 0


def test_span_growth_blocks_chained_merge():
    # A--B--C placed so A-B and B-C are legal but A-C exceeds the
    # tolerance; after A and B merge, C may not join (complete linkage).
    spectra = [
        [(100.0, 100.0)],
        [(100.004, 100.0)],
        [(100.008, 100.0)],
    ]
    # tolerance_da = 0.005: adjacent gaps legal, full span 0.008 illegal.
    result = build_consensus_spectrum(spectra, 0.005, 0.0, 2)
    assert all(p.replicate_count == 2 for p in result.peaks)
    assert len(result.peaks) == 1


def test_deterministic_tie_breaking():
    # Two disjoint pairs with identical span; the lower-m/z pair merges
    # first, and the result is stable across repeated runs.
    spectra = [
        [(100.0, 100.0), (200.0, 100.0)],
        [(100.002, 100.0), (200.002, 100.0)],
    ]
    first = build_consensus_spectrum(spectra, 0.01, 0.0, 2)
    second = build_consensus_spectrum(spectra, 0.01, 0.0, 2)
    sig = lambda res: [
        (
            round(p.mz, 9),
            tuple((m.spectrum, m.peak_index) for m in p.members),
        )
        for p in res.peaks
    ]
    assert sig(first) == sig(second)
    assert [round(p.mz, 4) for p in first.peaks] == [100.001, 200.001]


def test_result_is_deterministic_over_random_input():
    import random

    random.seed(123)
    bulk = []
    for _ in range(8):
        peaks = sorted(
            (100.0 + k * 1.0 + random.uniform(-0.003, 0.003),
             1000.0 * random.uniform(0.05, 1.0))
            for k in range(30)
        )
        bulk.append(peaks)
    r1 = build_consensus_spectrum(bulk, 0.01, 5.0, 3)
    r2 = build_consensus_spectrum(bulk, 0.01, 5.0, 3)
    sig = lambda res: tuple(
        (round(p.mz, 9), tuple((m.spectrum, m.peak_index) for m in p.members))
        for p in res.peaks
    )
    assert sig(r1) == sig(r2)
    assert all(p.replicate_count >= 3 for p in r1.peaks)
    assert [p.mz for p in r1.peaks] == sorted(p.mz for p in r1.peaks)


def test_zero_intensity_spectrum_rejected():
    spectra = [[(100.0, 0.0)], [(100.0, 1.0)]]
    with pytest.raises(ConsensusError) as exc:
        build_consensus_spectrum(spectra, 0.01, 5.0, 2)
    assert exc.value.error == "zero_spectrum"
    assert exc.value.spectrum == 0


def test_tight_tolerance_no_merges():
    spectra = [
        [(100.0, 1.0)],
        [(100.1, 1.0)],
    ]
    assert build_consensus_spectrum(spectra, 0.001, 1.0, 2).peaks == ()
    singles = build_consensus_spectrum(spectra, 0.001, 1.0, 1)
    assert len(singles.peaks) == 2


def test_weighted_mz_with_zero_intensity_member():
    # A zero-intensity member joins via span but must not move the m/z
    # numerator; the mean falls back to intensity-weighted contributors.
    spectra = [
        [(100.0, 100.0)],
        [(100.002, 0.0), (200.0, 1.0)],
    ]
    result = build_consensus_spectrum(spectra, 0.01, 5.0, 2)
    near100 = result.peaks[0]
    assert near100.mz == pytest.approx(100.0)
    # Normalised: spectrum 1's base is at 200, so the 100.002 member is 0.
    values = [100.0, 0.0]
    from statistics import median as med

    assert near100.median_intensity == pytest.approx(med(values))


# --- API endpoint -----------------------------------------------------------

def test_endpoint_consensus(client):
    payload = _payload(
        [
            [(100.0, 1000.0), (200.0, 500.0)],
            [(100.002, 500.0), (200.001, 1000.0)],
            [(100.004, 200.0), (110.2, 50.0), (200.0, 800.0)],
        ]
    )
    resp = client.post("/api/v1/consensus-spectrum", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["spectrum_count"] == 3
    assert body["total_peak_count"] == 7
    assert body["peak_count"] == 2
    assert body["min_replicates"] == 2
    first = body["peaks"][0]
    assert first["members"] == [
        {"spectrum": 0, "peak_index": 0},
        {"spectrum": 1, "peak_index": 0},
        {"spectrum": 2, "peak_index": 0},
    ]
    assert first["replicate_count"] == 3
    assert first["median_intensity"] == pytest.approx(50.0)
    assert body["peaks"] == sorted(body["peaks"], key=lambda p: p["mz"])


def test_endpoint_members_sorted_and_complete(client):
    payload = _payload(
        [
            [(100.0, 100.0), (100.001, 50.0), (200.0, 20.0)],
            [(100.0005, 100.0), (200.0005, 40.0)],
        ]
    )
    resp = client.post("/api/v1/consensus-spectrum", json=payload)
    assert resp.status_code == 200
    for peak in resp.json()["peaks"]:
        spectra_used = [m["spectrum"] for m in peak["members"]]
        assert spectra_used == sorted(spectra_used)
        assert len(spectra_used) == len(set(spectra_used))


def test_endpoint_non_increasing_mz_422(client):
    payload = _payload(
        [
            [(101.0, 1.0), (100.0, 1.0)],
            [(100.5, 1.0)],
        ]
    )
    resp = client.post("/api/v1/consensus-spectrum", json=payload)
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["error"] == "invalid_peaks"
    assert detail["spectrum"] == 0
    assert detail["index"] == 1


def test_endpoint_duplicate_mz_422(client):
    payload = _payload(
        [
            [(100.0, 1.0)],
            [(100.0, 1.0), (100.0, 1.0)],
        ]
    )
    resp = client.post("/api/v1/consensus-spectrum", json=payload)
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["error"] == "invalid_peaks"
    assert detail["spectrum"] == 1
    assert detail["index"] == 1


def test_endpoint_zero_intensity_spectrum_422(client):
    payload = _payload(
        [
            [(100.0, 1.0)],
            [(100.0, 0.0)],
        ]
    )
    resp = client.post("/api/v1/consensus-spectrum", json=payload)
    assert resp.status_code == 422
    assert resp.json()["detail"]["error"] == "zero_spectrum"
    assert resp.json()["detail"]["spectrum"] == 1


def test_endpoint_negative_intensity_422(client):
    payload = _payload([[(100.0, 1.0)], [(100.0, -1.0)]])
    assert client.post("/api/v1/consensus-spectrum", json=payload).status_code == 422


def test_endpoint_parameter_bounds(client):
    good = _payload([[(100.0, 1.0)], [(100.0, 1.0)]])
    for overrides in (
        {"tolerance_da": 0.0},
        {"tolerance_da": -1.0},
        {"tolerance_ppm": 0.0},
        {"tolerance_ppm": -2.0},
        {"min_replicates": 0},
        {"min_replicates": 21},
        {"min_replicates": 3},
    ):
        payload = {**good, **overrides}
        resp = client.post("/api/v1/consensus-spectrum", json=payload)
        assert resp.status_code == 422, overrides


def test_endpoint_spectrum_count_bounds(client):
    one = _payload([[(100.0, 1.0)]])
    assert client.post("/api/v1/consensus-spectrum", json=one).status_code == 422
    many = _payload([[(100.0, 1.0)] for _ in range(21)])
    assert client.post("/api/v1/consensus-spectrum", json=many).status_code == 422


def test_endpoint_peak_count_bounds(client):
    empty = _payload([[], [(100.0, 1.0)]])
    assert client.post("/api/v1/consensus-spectrum", json=empty).status_code == 422
    too_many = _payload(
        [[(float(i), 1.0) for i in range(201)], [(100.0, 1.0)]]
    )
    assert client.post("/api/v1/consensus-spectrum", json=too_many).status_code == 422


def test_endpoint_non_finite_tolerances_422(client):
    for raw in (
        '{"tolerance_da":NaN,"tolerance_ppm":5.0,"min_replicates":2,'
        '"spectra":[{"peaks":[{"mz":100.0,"intensity":1.0}]},'
        '{"peaks":[{"mz":100.0,"intensity":1.0}]}]}',
        '{"tolerance_da":0.01,"tolerance_ppm":Infinity,"min_replicates":2,'
        '"spectra":[{"peaks":[{"mz":100.0,"intensity":1.0}]},'
        '{"peaks":[{"mz":100.0,"intensity":1.0}]}]}',
    ):
        resp = client.post(
            "/api/v1/consensus-spectrum",
            content=raw,
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422, raw


def test_endpoint_min_replicates_greater_than_spectra_422(client):
    payload = _payload(
        [[(100.0, 1.0)], [(100.0, 1.0)]], min_replicates=3
    )
    resp = client.post("/api/v1/consensus-spectrum", json=payload)
    assert resp.status_code == 422
    assert resp.json()["detail"]["error"] == "invalid_min_replicates"


def test_endpoint_no_consensus_peaks_is_empty_200(client):
    payload = _payload([[(100.0, 1.0)], [(200.0, 1.0)]])
    resp = client.post("/api/v1/consensus-spectrum", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["peak_count"] == 0
    assert body["peaks"] == []
