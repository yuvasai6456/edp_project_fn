from app import app


def test_health_endpoint():
    client = app.test_client()
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"


def test_homepage_loads():
    client = app.test_client()
    response = client.get("/")
    assert response.status_code == 200
    assert b"Credit Card Fraud Detector" in response.data


def test_invalid_hour_is_rejected():
    client = app.test_client()
    response = client.post("/", data={
        "transaction_id": "10001",
        "amount": "100",
        "transaction_hour": "25",
        "merchant_category": "Travel",
        "foreign_transaction": "0",
        "location_mismatch": "0",
        "device_trust_score": "90",
        "velocity_last_24h": "1",
        "cardholder_age": "35",
    })
    assert response.status_code == 200
    assert b"Transaction hour must be between 0 and 23." in response.data


def test_saved_model_can_predict():
    from src.predict import predict_fraud

    sample = {
        "transaction_id": 10001,
        "amount": 8500.0,
        "transaction_hour": 2,
        "merchant_category": "Travel",
        "foreign_transaction": 1,
        "location_mismatch": 1,
        "device_trust_score": 25.0,
        "velocity_last_24h": 8,
        "cardholder_age": 35,
    }
    result = predict_fraud(sample)
    assert result["prediction"] in {"Fraud", "Legitimate"}
    assert 0.0 <= result["fraud_probability"] <= 1.0
    assert result["model_version"]
