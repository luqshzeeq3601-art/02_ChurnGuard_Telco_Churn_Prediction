"""Latency benchmark suite for FastAPI /predict endpoint (T5.5 / EO2).

Verifies that p95 latency for single-customer prediction is strictly under 100 ms.
"""

import time

import numpy as np
import pytest
from api.main import app
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create TestClient instance."""
    return TestClient(app)


@pytest.fixture
def sample_payload():
    """Standard customer payload for benchmarking."""
    return {
        "customerID": "7590-VHVEG",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85,
    }


def run_latency_benchmark(
    client: TestClient, payload: dict, n_requests: int = 100
) -> dict[str, float]:
    """Execute n_requests sequential calls to /predict and return latency statistics in milliseconds."""
    # Warmup
    for _ in range(5):
        client.post("/predict", json=payload)

    latencies_ms = []
    for _ in range(n_requests):
        t0 = time.perf_counter()
        resp = client.post("/predict", json=payload)
        t1 = time.perf_counter()
        assert resp.status_code == 200
        latencies_ms.append((t1 - t0) * 1000.0)

    arr = np.array(latencies_ms)
    stats = {
        "n_requests": n_requests,
        "mean_ms": float(np.mean(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "max_ms": float(np.max(arr)),
        "min_ms": float(np.min(arr)),
    }
    return stats


def test_predict_latency_p95_under_100ms(client, sample_payload):
    """Verify EO2 requirement: /predict p95 latency under 100 ms locally."""
    stats = run_latency_benchmark(client, sample_payload, n_requests=100)

    print("\n" + "=" * 60)
    print("LATENCY BENCHMARK RESULTS (100 Requests to POST /predict)")
    print("=" * 60)
    print(f"  Mean Latency:    {stats['mean_ms']:.2f} ms")
    print(f"  Median (p50):    {stats['p50_ms']:.2f} ms")
    print(f"  90th Percentile: {stats['p90_ms']:.2f} ms")
    print(f"  95th Percentile: {stats['p95_ms']:.2f} ms  (Target: < 100 ms)")
    print(f"  99th Percentile: {stats['p99_ms']:.2f} ms")
    print(f"  Min / Max:       {stats['min_ms']:.2f} ms / {stats['max_ms']:.2f} ms")
    print("=" * 60)

    assert stats["p95_ms"] < 100.0, f"p95 latency {stats['p95_ms']:.2f} ms exceeded 100 ms target!"


if __name__ == "__main__":
    with TestClient(app) as test_client:
        sample = {
            "customerID": "7590-VHVEG",
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 1,
            "PhoneService": "No",
            "MultipleLines": "No phone service",
            "InternetService": "DSL",
            "OnlineSecurity": "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 29.85,
            "TotalCharges": 29.85,
        }
        res = run_latency_benchmark(test_client, sample, n_requests=100)
        print(f"Benchmark: Mean = {res['mean_ms']:.2f} ms, p95 = {res['p95_ms']:.2f} ms")
