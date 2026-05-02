from __future__ import annotations

import json
import shutil
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

import pytest


@dataclass(frozen=True)
class HttpieResponse:
    status_code: int
    headers: dict[str, str]
    body: dict[str, Any]
    returncode: int
    stderr: str


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture(scope="session")
def httpie_bin() -> str:
    binary = shutil.which("http")
    if binary is None:
        pytest.skip("HTTPie CLI (`http`) is not installed in this environment.")
    return binary


@pytest.fixture(scope="session")
def base_url() -> Iterator[str]:
    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "tests.e2e.httpie_app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if process.poll() is not None:
            _, stderr = process.communicate(timeout=1)
            pytest.fail(f"E2E API server exited before startup:\n{stderr}")
        try:
            with urlopen(f"{url}/health", timeout=0.5) as response:
                if response.status == 200:
                    break
        except URLError:
            time.sleep(0.1)
    else:
        process.terminate()
        _, stderr = process.communicate(timeout=5)
        pytest.fail(f"E2E API server did not become ready:\n{stderr}")

    yield url

    process.terminate()
    try:
        process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate(timeout=5)


def _run_httpie(
    httpie_bin: str,
    *args: str,
    expected_status: int,
) -> HttpieResponse:
    completed = subprocess.run(
        [
            httpie_bin,
            "--check-status",
            "--ignore-stdin",
            "--print=hb",
            "--pretty=none",
            "--timeout=5",
            *args,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    header_text, _, body_text = completed.stdout.partition("\n\n")
    status_line, *header_lines = header_text.splitlines()
    status_code = int(status_line.split()[1])
    headers = {
        key.lower(): value.strip()
        for line in header_lines
        if ": " in line
        for key, value in [line.split(": ", 1)]
    }
    body = json.loads(body_text) if body_text.strip() else {}

    assert status_code == expected_status, completed.stdout + completed.stderr
    return HttpieResponse(
        status_code=status_code,
        headers=headers,
        body=body,
        returncode=completed.returncode,
        stderr=completed.stderr,
    )


def _valid_predict_args(**overrides: str) -> list[str]:
    fields = {
        "gender": "Female",
        "SeniorCitizen:": "0",
        "Partner": "Yes",
        "Dependents": "No",
        "tenure:": "24",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "One year",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges:": "75.5",
        "TotalCharges:": "1850.0",
        **overrides,
    }
    return [f"{key}={value}" for key, value in fields.items()]


def test_health_endpoint_with_query_param_via_httpie(
    httpie_bin: str, base_url: str
) -> None:
    response = _run_httpie(
        httpie_bin,
        "GET",
        f"{base_url}/health",
        "source==httpie-e2e",
        expected_status=200,
    )

    assert response.body["status"] == "ok"
    assert response.body["timestamp"].endswith("Z")
    assert "x-process-time" in response.headers
    assert "x-request-id" in response.headers


def test_predict_accepts_valid_payload_and_propagates_request_id(
    httpie_bin: str, base_url: str
) -> None:
    response = _run_httpie(
        httpie_bin,
        "POST",
        f"{base_url}/predict",
        "X-Request-ID:e2e-httpie-123",
        *_valid_predict_args(Contract="Month-to-month"),
        expected_status=200,
    )

    assert response.headers["x-request-id"] == "e2e-httpie-123"
    assert response.body == {
        "churn_probability": 0.91,
        "prediction": True,
        "threshold": 0.20303030303030303,
        "model_version": "e2e-test-version",
        "request_id": "e2e-httpie-123",
    }


@pytest.mark.parametrize(
    ("case", "overrides"),
    [
        ("new_customer_zero_tenure", {"tenure:": "0", "TotalCharges": " "}),
        ("max_tenure", {"tenure:": "120", "TotalCharges:": "9000.0"}),
        ("no_monthly_charge", {"MonthlyCharges:": "0", "TotalCharges:": "0"}),
        (
            "no_internet_service",
            {
                "InternetService": "No",
                "OnlineSecurity": "No internet service",
                "OnlineBackup": "No internet service",
                "DeviceProtection": "No internet service",
                "TechSupport": "No internet service",
                "StreamingTV": "No internet service",
                "StreamingMovies": "No internet service",
            },
        ),
        (
            "no_phone_service",
            {"PhoneService": "No", "MultipleLines": "No phone service"},
        ),
    ],
)
def test_predict_accepts_common_boundary_payloads(
    httpie_bin: str,
    base_url: str,
    case: str,
    overrides: dict[str, str],
) -> None:
    response = _run_httpie(
        httpie_bin,
        "POST",
        f"{base_url}/predict",
        f"X-Request-ID:{case}",
        *_valid_predict_args(**overrides),
        expected_status=200,
    )

    assert response.body["request_id"] == case
    assert 0 <= response.body["churn_probability"] <= 1
    assert isinstance(response.body["prediction"], bool)


@pytest.mark.parametrize(
    ("case", "overrides", "expected_field"),
    [
        ("invalid_enum", {"Contract": "Daily"}, "Contract"),
        ("negative_tenure", {"tenure:": "-1"}, "tenure"),
        ("tenure_above_limit", {"tenure:": "121"}, "tenure"),
        ("invalid_senior_citizen", {"SeniorCitizen:": "2"}, "SeniorCitizen"),
        ("negative_monthly_charge", {"MonthlyCharges:": "-0.01"}, "MonthlyCharges"),
        ("wrong_type_for_tenure", {"tenure": "twenty-four"}, "tenure"),
    ],
)
def test_predict_rejects_invalid_parameters(
    httpie_bin: str,
    base_url: str,
    case: str,
    overrides: dict[str, str],
    expected_field: str,
) -> None:
    response = _run_httpie(
        httpie_bin,
        "POST",
        f"{base_url}/predict",
        f"X-Request-ID:{case}",
        *_valid_predict_args(**overrides),
        expected_status=422,
    )

    assert response.returncode == 4
    assert any(item["loc"][-1] == expected_field for item in response.body["detail"])


def test_predict_rejects_missing_required_parameter(
    httpie_bin: str, base_url: str
) -> None:
    args = _valid_predict_args()
    args.remove("gender=Female")

    response = _run_httpie(
        httpie_bin,
        "POST",
        f"{base_url}/predict",
        *args,
        expected_status=422,
    )

    assert any(item["loc"][-1] == "gender" for item in response.body["detail"])


def test_predict_rejects_malformed_json_body(httpie_bin: str, base_url: str) -> None:
    response = _run_httpie(
        httpie_bin,
        "POST",
        f"{base_url}/predict",
        "Content-Type:application/json",
        "--raw",
        '{"gender":',
        expected_status=422,
    )

    assert response.returncode == 4
    assert response.body["detail"][0]["type"] == "json_invalid"


def test_predict_rejects_get_method(httpie_bin: str, base_url: str) -> None:
    response = _run_httpie(
        httpie_bin,
        "GET",
        f"{base_url}/predict",
        expected_status=405,
    )

    assert response.returncode == 4
    assert response.body["detail"] == "Method Not Allowed"


def test_unknown_endpoint_returns_404(httpie_bin: str, base_url: str) -> None:
    response = _run_httpie(
        httpie_bin,
        "GET",
        f"{base_url}/not-found",
        expected_status=404,
    )

    assert response.returncode == 4
    assert response.body["detail"] == "Not Found"


@pytest.mark.xfail(
    reason=(
        "PredictRequest currently ignores unknown JSON fields. Consider "
        "ConfigDict(extra='forbid') to reject misspelled or unsupported params."
    ),
    strict=True,
)
def test_predict_rejects_unknown_parameter(httpie_bin: str, base_url: str) -> None:
    response = _run_httpie(
        httpie_bin,
        "POST",
        f"{base_url}/predict",
        *_valid_predict_args(customerID="7590-VHVEG"),
        expected_status=422,
    )

    assert any(item["loc"][-1] == "customerID" for item in response.body["detail"])
