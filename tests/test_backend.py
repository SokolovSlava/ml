import pytest
from ml.backend.app import app, IrisType
from fastapi.testclient import TestClient

@pytest.fixture
def client() -> TestClient:
    return TestClient(app)

def test_healthcheck(client: TestClient):
    #response = client.get("/healthcheck")
    response = client.request("GET", "/healthcheck")
    assert response.status_code == 200

def test_predict(client: TestClient):
    response = client.request(
        "GET",
        "/predict",
        json={
            "petal_height": 10,
            "petal_width": 10,
            "sepal_height": 10,
            "sepal_width": 10,
        },
    )
    #assert response.status_code == 200
    assert IrisType(response.json()["predict"])

"""

def sum_of_numbers(a: int, b: int):
    return a + b

@pytest.mark.parametrize(
    ["a", "b"],
    [
        [10, 10],
        [0, 5],
        [1, 1]

    ]
)
def test_sum_of_numbers(a: int, b: int):
    result = sum_of_numbers(a=a, b=b)
    assert result == (a + b)
    
"""

