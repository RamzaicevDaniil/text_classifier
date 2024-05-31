from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_predict():
    response = client.post("/predict", json={"text": "some sample text"})
    assert response.status_code == 200
    assert "prediction" in response.json()


def test_predict_batch():
    response = client.post("/predict_batch", json={"texts": ["some sample text", "another text"]})
    assert response.status_code == 200
    assert "predictions" in response.json()
    assert len(response.json()["predictions"]) == 2