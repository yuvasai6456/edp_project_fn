import logging
import os

from flask import Flask, jsonify, render_template, request
from src.predict import predict_fraud

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024  # small form-only request limit

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


@app.get("/healthz")
def healthz():
    """Health check used by Render."""
    return jsonify({"status": "ok"}), 200


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None

    if request.method == "POST":
        try:
            transaction_id = int(request.form["transaction_id"])
            amount = float(request.form["amount"])
            transaction_hour = int(request.form["transaction_hour"])
            merchant_category = request.form["merchant_category"].strip()
            foreign_transaction = int(request.form["foreign_transaction"])
            location_mismatch = int(request.form["location_mismatch"])
            device_trust_score = float(request.form["device_trust_score"])
            velocity_last_24h = int(request.form["velocity_last_24h"])
            cardholder_age = int(request.form["cardholder_age"])

            if transaction_id < 0:
                raise ValueError("Transaction ID must be zero or greater.")
            if amount < 0:
                raise ValueError("Amount cannot be negative.")
            if not 0 <= transaction_hour <= 23:
                raise ValueError("Transaction hour must be between 0 and 23.")
            if not merchant_category:
                raise ValueError("Merchant category is required.")
            if foreign_transaction not in (0, 1) or location_mismatch not in (0, 1):
                raise ValueError("Transaction flags must be either Yes or No.")
            if not 0 <= device_trust_score <= 100:
                raise ValueError("Device trust score must be between 0 and 100.")
            if velocity_last_24h < 0:
                raise ValueError("Transaction velocity cannot be negative.")
            if not 18 <= cardholder_age <= 120:
                raise ValueError("Cardholder age must be between 18 and 120.")

            transaction = {
                "transaction_id": transaction_id,
                "amount": amount,
                "transaction_hour": transaction_hour,
                "merchant_category": merchant_category,
                "foreign_transaction": foreign_transaction,
                "location_mismatch": location_mismatch,
                "device_trust_score": device_trust_score,
                "velocity_last_24h": velocity_last_24h,
                "cardholder_age": cardholder_age,
            }

            result = predict_fraud(transaction)

            # Log operational metrics only; do not log transaction identifiers
            # or raw transaction fields.
            logger.info(
                "prediction_completed model_version=%s prediction=%s fraud_probability=%.4f",
                result.get("model_version", "unknown"),
                result["prediction"],
                result["fraud_probability"],
            )

        except (ValueError, KeyError, TypeError) as exc:
            error = str(exc)
        except Exception:
            logger.exception("Prediction failed")
            error = "Prediction could not be completed. Please verify the inputs and try again."

    return render_template("index.html", result=result, error=error)


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "5000")),
        debug=False,
    )
