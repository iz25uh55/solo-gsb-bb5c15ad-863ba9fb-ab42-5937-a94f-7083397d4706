"""Tests for the replicate consensus-spectrum builder and its API endpoint."""

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.consensus import ConsensusError, build_consensus_spectrum


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "db_path", str(tmp_path / "test.db"))
    from app.main import app

    with TestClient(app) as c:
        yield c


def _payload(spectra, tolerance_da=0.005, tolerance_ppm=0.0, min_replicates=2):
    return {
        "spectra": [
            {"peaks": [{"mz": m, "intensity": i} for m, i in peaks]}
            for peaks in spectra
        ],
        "tolerance_da": tolerance_da,
        "tolerance_ppm": tolerance_ppm,
        "min_replicates": min_replicates,
    }


def _members(cluster):
    return [(m.spectrum_index, m.peak_index) for m in cluster.members]


# --- unit-level clustering ----------------------------------------------------

def test_basic_merge_and_statistics():
    spectra = [
        [(100.0, 50.0), (200.0, 100.0)],
        [(100.004, 80.0), (200.006, 40.0)],
        [(100.002, 30.0)],
    ]
    result = build_consensus_spectrum(spectra, 0.005, 0.0, 2)
    assert result.spectrum_count == 3
    assert result.peak_count == 5
    # The 200 Da peaks are 0.006 apart (beyond tolerance) and stay singletons.
    assert len(result.clusters) == 1
    cluster = result.clusters[0]
    # Original-intensity weighting: (100*50 + 100.002*30 + 100.004*80) / 160.
    assert cluster.mz == pytest.approx(100.002375)
    # Normalised member intensities 50, 100, 100 -> median 100.
    assert cluster.intensity == pytest.approx(100.0)
    assert _members(cluster) == [(0, 0), (1, 0), (2, 0)]


def test_ppm_tolerance_widens_span_at_high_mz():
    spectra = [[(1000.0, 100.0)], [(1000.004, 50.0)]]
    # Da alone (0.001) forbids the 0.004 span; 5 ppm allows ~0.005.
    assert len(build_consensus_spectrum(spectra, 0.001, 5.0, 2).clusters) == 1
    # 1 ppm allows only ~0.001 at this m/z: no merge.
    assert build_consensus_spectrum(spectra, 0.001, 1.0, 2).clusters == ()


def test_same_spectrum_peaks_never_merge():
    spectra = [
        [(100.0, 100.0), (100.003, 90.0)],
        [(100.001, 80.0)],
    ]
    result = build_consensus_spectrum(spectra, 0.01, 0.0, 2)
    # The two spectrum-0 peaks can never share a cluster; the closer one
    # pairs with the spectrum-1 peak, the other stays a singleton.
    assert len(result.clusters) == 1
    cluster = result.clusters[0]
    assert _members(cluster) == [(0, 0), (1, 0)]
    assert cluster.mz == pytest.approx((100.0 * 100 + 100.001 * 80) / 180)
    assert cluster.intensity == pytest.approx(100.0)


def test_complete_linkage_span_blocks_chaining():
    # Exactly representable 1/128 Da steps: both adjacent spans tie at
    # 0.0078125, so the lowest-m/z pair merges first; the merged 0.015625
    # span then exceeds the tolerance and the chain cannot grow.
    spectra = [
        [(100.0, 100.0)],
        [(100.0078125, 100.0)],
        [(100.015625, 100.0)],
    ]
    result = build_consensus_spectrum(spectra, 0.008, 0.0, 2)
    assert len(result.clusters) == 1
    cluster = result.clusters[0]
    assert _members(cluster) == [(0, 0), (1, 0)]
    assert cluster.mz == pytest.approx(100.00390625)
    # Median of normalised [100, 100, 0] over all three spectra.
    assert cluster.intensity == pytest.approx(100.0)


def test_non_adjacent_merge():
    # Round 1 merges the closest pair (100.04, 100.05); afterwards the only
    # legal pair is the non-adjacent (100.0, 100.09) across it.
    spectra = [
        [(100.0, 10.0), (100.04, 20.0)],
        [(100.05, 30.0), (100.09, 40.0)],
    ]
    result = build_consensus_spectrum(spectra, 0.1, 0.0, 2)
    assert len(result.clusters) == 2
    first, second = result.clusters
    assert _members(first) == [(0, 1), (1, 0)]
    assert first.mz == pytest.approx((100.04 * 20 + 100.05 * 30) / 50)
    # Normalised: 100.04 -> 100 (base of spectrum 0), 100.05 -> 75.
    assert first.intensity == pytest.approx(87.5)
    assert _members(second) == [(0, 0), (1, 1)]
    assert second.mz == pytest.approx((100.0 * 10 + 100.09 * 40) / 50)
    # Normalised: 100.0 -> 50, 100.09 -> 100.
    assert second.intensity == pytest.approx(75.0)


def test_min_replicates_filters_clusters():
    spectra = [[(100.0, 100.0)], [(100.001, 100.0)], [(500.0, 100.0)]]
    assert build_consensus_spectrum(spectra, 0.01, 0.0, 3).clusters == ()
    result = build_consensus_spectrum(spectra, 0.01, 0.0, 2)
    assert len(result.clusters) == 1
    assert _members(result.clusters[0]) == [(0, 0), (1, 0)]


def test_median_counts_missing_spectra_as_zero():
    spectra = [
        [(100.0, 100.0)],
        [(100.001, 60.0), (150.0, 100.0)],
        [(300.0, 100.0)],
        [(400.0, 100.0)],
    ]
    result = build_consensus_spectrum(spectra, 0.01, 0.0, 2)
    assert len(result.clusters) == 1
    cluster = result.clusters[0]
    # Normalised values over all four spectra: [100, 60, 0, 0] -> median 30.
    assert cluster.intensity == pytest.approx(30.0)
    assert cluster.mz == pytest.approx((100.0 * 100 + 100.001 * 60) / 160)


def test_zero_weight_cluster_falls_back_to_plain_mean():
    spectra = [
        [(100.0, 0.0), (200.0, 5.0)],
        [(100.001, 0.0), (300.0, 7.0)],
    ]
    result = build_consensus_spectrum(spectra, 0.01, 0.0, 2)
    assert len(result.clusters) == 1
    cluster = result.clusters[0]
    assert cluster.mz == pytest.approx(100.0005)
    assert cluster.intensity == pytest.approx(0.0)


def test_identical_mz_across_spectra_merge():
    spectra = [[(100.0, 100.0)], [(100.0, 50.0)], [(100.0, 25.0)]]
    result = build_consensus_spectrum(spectra, 0.001, 0.0, 2)
    assert len(result.clusters) == 1
    cluster = result.clusters[0]
    assert _members(cluster) == [(0, 0), (1, 0), (2, 0)]
    assert cluster.mz == pytest.approx(100.0)
    assert cluster.intensity == pytest.approx(100.0)


def test_deterministic_repeated_runs():
    spectra = [
        [(100.0, 10.0), (100.04, 20.0), (150.0, 5.0)],
        [(100.05, 30.0), (100.09, 40.0), (150.003, 2.0)],
        [(100.02, 7.0), (200.0, 3.0)],
    ]
    first = build_consensus_spectrum(spectra, 0.1, 5.0, 2)
    second = build_consensus_spectrum(spectra, 0.1, 5.0, 2)
    assert first == second


def test_non_increasing_mz_rejected():
    with pytest.raises(ConsensusError) as excinfo:
        build_consensus_spectrum(
            [[(100.0, 1.0), (99.0, 2.0)], [(100.0, 1.0)]], 0.01, 0.0, 2
        )
    assert excinfo.value.error == "invalid_peaks"
    assert excinfo.value.spectrum_index == 0
    assert excinfo.value.peak_index == 1


def test_duplicate_mz_rejected():
    with pytest.raises(ConsensusError) as excinfo:
        build_consensus_spectrum(
            [[(100.0, 1.0)], [(50.0, 1.0), (50.0, 2.0)]], 0.01, 0.0, 2
        )
    assert excinfo.value.error == "invalid_peaks"
    assert excinfo.value.spectrum_index == 1
    assert excinfo.value.peak_index == 1


def test_all_zero_spectrum_rejected():
    with pytest.raises(ConsensusError) as excinfo:
        build_consensus_spectrum(
            [[(100.0, 0.0), (101.0, 0.0)], [(100.0, 1.0)]], 0.01, 0.0, 2
        )
    assert excinfo.value.error == "zero_spectrum"
    assert excinfo.value.spectrum_index == 0
    assert excinfo.value.peak_index is None


# --- API endpoint ---------------------------------------------------------------

def test_endpoint_happy_path(client):
    resp = client.post(
        "/api/v1/consensus-spectrum",
        json=_payload(
            [
                [(100.0, 50.0), (200.0, 100.0)],
                [(100.004, 80.0), (200.006, 40.0)],
                [(100.002, 30.0)],
            ]
        ),
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["spectrum_count"] == 3
    assert body["tolerance_da"] == 0.005
    assert body["tolerance_ppm"] == 0.0
    assert body["min_replicates"] == 2
    assert body["peak_count"] == 5
    assert body["cluster_count"] == 1
    cluster = body["clusters"][0]
    assert cluster["mz"] == pytest.approx(100.002375, abs=1e-6)
    assert cluster["intensity"] == pytest.approx(100.0)
    assert cluster["spectrum_count"] == 3
    assert cluster["members"] == [
        {"spectrum_index": 0, "peak_index": 0},
        {"spectrum_index": 1, "peak_index": 0},
        {"spectrum_index": 2, "peak_index": 0},
    ]


def test_endpoint_clusters_sorted_by_mz(client):
    resp = client.post(
        "/api/v1/consensus-spectrum",
        json=_payload(
            [
                [(100.0, 10.0), (100.04, 20.0)],
                [(100.05, 30.0), (100.09, 40.0)],
            ],
            tolerance_da=0.1,
        ),
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["cluster_count"] == 2
    mzs = [cluster["mz"] for cluster in body["clusters"]]
    assert mzs == sorted(mzs)
    assert mzs[0] == pytest.approx(100.046, abs=1e-6)
    assert mzs[1] == pytest.approx(100.072, abs=1e-6)


def test_endpoint_empty_when_threshold_unreachable(client):
    resp = client.post(
        "/api/v1/consensus-spectrum",
        json=_payload([[(100.0, 1.0)], [(100.001, 1.0)]], min_replicates=3),
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["clusters"] == []
    assert body["cluster_count"] == 0


def test_endpoint_spectrum_count_bounds(client):
    one = _payload([[(100.0, 1.0)]])
    assert client.post("/api/v1/consensus-spectrum", json=one).status_code == 422
    twenty_one = _payload([[(100.0, 1.0)]] * 21)
    assert (
        client.post("/api/v1/consensus-spectrum", json=twenty_one).status_code
        == 422
    )


def test_endpoint_peak_count_bounds(client):
    empty = _payload([[], [(100.0, 1.0)]])
    assert client.post("/api/v1/consensus-spectrum", json=empty).status_code == 422
    too_many = _payload(
        [[(float(i), 1.0) for i in range(201)], [(100.0, 1.0)]]
    )
    assert (
        client.post("/api/v1/consensus-spectrum", json=too_many).status_code
        == 422
    )


def test_endpoint_parameter_bounds(client):
    base = _payload([[(100.0, 1.0)], [(100.001, 1.0)]])
    for overrides in (
        {"tolerance_da": 0.0},
        {"tolerance_da": -0.1},
        {"tolerance_ppm": -1.0},
        {"min_replicates": 1},
        {"min_replicates": 21},
    ):
        payload = {**base, **overrides}
        resp = client.post("/api/v1/consensus-spectrum", json=payload)
        assert resp.status_code == 422, overrides


def test_endpoint_negative_intensity_422(client):
    payload = _payload([[(100.0, 1.0)], [(100.001, -1.0)]])
    assert client.post("/api/v1/consensus-spectrum", json=payload).status_code == 422


def test_endpoint_non_increasing_mz_422(client):
    resp = client.post(
        "/api/v1/consensus-spectrum",
        json=_payload([[(100.0, 1.0)], [(101.0, 1.0), (100.0, 2.0)]]),
    )
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["error"] == "invalid_peaks"
    assert detail["spectrum_index"] == 1
    assert detail["peak_index"] == 1


def test_endpoint_all_zero_spectrum_422(client):
    resp = client.post(
        "/api/v1/consensus-spectrum",
        json=_payload([[(100.0, 0.0)], [(100.0, 1.0)]]),
    )
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["error"] == "zero_spectrum"
    assert detail["spectrum_index"] == 0


def test_endpoint_non_finite_values_422(client):
    # httpx refuses to emit NaN/Infinity itself, so send raw JSON text; the
    # service parses these Python-flavoured literals and must reject them.
    head = (
        '{"spectra":[{"peaks":[{"mz":100.0,"intensity":1.0}]},'
        '{"peaks":[{"mz":100.001,"intensity":1.0}]}],'
    )
    tail = '"tolerance_da":0.005,"tolerance_ppm":0.0,"min_replicates":2}'
    cases = [
        # Non-finite mz / intensity inside a spectrum.
        head.replace('{"mz":100.001,"intensity":1.0}', '{"mz":NaN,"intensity":1.0}') + tail,
        head.replace('{"mz":100.001,"intensity":1.0}', '{"mz":Infinity,"intensity":1.0}') + tail,
        head.replace('{"mz":100.001,"intensity":1.0}', '{"mz":100.001,"intensity":NaN}') + tail,
        head.replace('{"mz":100.001,"intensity":1.0}', '{"mz":100.001,"intensity":Infinity}') + tail,
        # Non-finite tolerances.
        head + '"tolerance_da":NaN,"tolerance_ppm":0.0,"min_replicates":2}',
        head + '"tolerance_da":Infinity,"tolerance_ppm":0.0,"min_replicates":2}',
        head + '"tolerance_da":0.005,"tolerance_ppm":Infinity,"min_replicates":2}',
    ]
    for body in cases:
        resp = client.post(
            "/api/v1/consensus-spectrum",
            content=body,
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422, body
