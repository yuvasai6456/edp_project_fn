# Credit Card Fraud Detector — Render Deployment + MLOps Mini Pipeline

This project preserves the existing Flask + HTML interface and serves the uploaded XGBoost model artifacts.

## 1. Project structure

```text
.
├── app.py
├── requirements.txt              # lightweight production dependencies
├── requirements-dev.txt          # training, MLflow and test dependencies
├── render.yaml                   # Render Blueprint configuration
├── .python-version               # Python 3.12.8
├── models/
│   ├── preprocessor.pkl
│   ├── xgb_classifier.json
│   ├── model_metadata.json
│   └── existing legacy .pkl files
├── src/
│   └── predict.py
├── scripts/
│   └── train.py                  # reproducible training + MLflow tracking
├── tests/
│   └── test_app.py
├── .github/workflows/ci.yml      # test then optional Render deploy hook
└── templates/
    └── index.html                # existing interface retained
```

## 2. Important model compatibility note

The notebook records scikit-learn 1.6.1, XGBoost 3.4.1 and imbalanced-learn 0.14.2. The serialized `preprocessor.pkl` failed to load under scikit-learn 1.8.0 in the inspection environment. Production therefore pins scikit-learn to 1.6.1. XGBoost's native JSON artifact is loaded separately.

The live prediction path uses `preprocessor.pkl` and `xgb_classifier.json`. The older `fraud_detection_model.pkl` files are retained for reference but are not loaded by the Flask app.

The notebook's separate threshold-analysis cell selected 0.40, while the later deployment artifact test cells used 0.50. The current web app keeps 0.50 to match the deployed artifact test. Change `FRAUD_THRESHOLD` only after evaluating that threshold against the exact exported model.

## 3. Run locally (Windows PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000.

For tests and model retraining tools:

```powershell
pip install -r requirements-dev.txt
pytest -q
```

## 4. Deploy to Render

1. Push this project to a GitHub repository. Do not push `.venv`, datasets, `.env` files, or credentials.
2. In Render, choose **New → Blueprint** (or create a Web Service connected to the repository).
3. Render reads `render.yaml`. Confirm the root directory is the repository root.
4. Confirm the Python runtime is 3.12.8 and the build/start commands match `render.yaml`.
5. Deploy. Render builds with `pip install -r requirements.txt` and starts Gunicorn.
6. Test both:
   - `https://YOUR-SERVICE.onrender.com/`
   - `https://YOUR-SERVICE.onrender.com/healthz`

The public URL is assigned by Render after deployment; it is not created by this ZIP.

### Render environment

The application reads the threshold from `models/model_metadata.json` (currently `0.50` for the existing artifacts). `FRAUD_THRESHOLD`, if explicitly set in Render, overrides the metadata value. `PORT` is provided by Render and is read automatically.

The free Render service may spin down after inactivity, so the first request after a quiet period can take longer. The service filesystem is ephemeral; the model artifacts are included in the repository and loaded at startup.

## 5. Enable CI/CD

The included GitHub Actions workflow runs tests on pushes and pull requests to `main`. The Render Blueprint uses `autoDeployTrigger: checksPass`, so Render waits for connected GitHub CI checks to pass before deploying the linked branch.

1. Push the project to GitHub.
2. In Render, create the service from the repository's `render.yaml` Blueprint.
3. Confirm the service is connected to the same GitHub repository and `main` branch.
4. Confirm Auto-Deploy is set to **After CI Checks Pass**.
5. Push a commit. GitHub Actions runs `pytest`; after successful checks, Render deploys.

No deploy-hook secret is required with this setup. See Render's [CI integration documentation](https://render.com/docs/deploys#integrating-with-ci).

## 6. MLOps training + MLflow experiment tracking

The uploaded ZIP does not include the training CSV, so the training command below is a template that requires your original dataset. It expects the nine input columns used by the model plus the binary target column `is_fraud`.

Place your dataset at `data/transactions.csv` (the `data/` folder is ignored by Git), then run:

```powershell
pip install -r requirements-dev.txt
python scripts/train.py --data data/transactions.csv --version 1.0.1
```

The script:
- Splits data into train, validation and test sets.
- Fits preprocessing and SMOTE only on training data through an imbalanced-learn pipeline.
- Selects the classification threshold on validation data.
- Evaluates once on the held-out test set.
- Exports `preprocessor.pkl`, `xgb_classifier.json` and `model_metadata.json`.
- Logs parameters, metrics and model artifacts to a local MLflow experiment.

View local MLflow runs:

```powershell
mlflow ui --backend-store-uri ./mlruns
```

Then open http://127.0.0.1:5000 (use another port if Flask is already running, e.g. `mlflow ui --port 5001`).

Run training locally or in a controlled training job, review the metrics, then commit/release the approved model artifacts. Do not train on the Render web-service instance.

## 7. Current MLOps coverage

| Capability | Status |
|---|---|
| Model training and evaluation notebook | Existing |
| Saved model and preprocessing artifacts | Existing |
| Flask model serving | Existing; deployment configuration added |
| Reproducible training script | Added; requires original dataset |
| MLflow experiment tracking | Added; requires training run |
| Model metadata/version label | Added for baseline artifact |
| Automated tests | Added; must pass in GitHub Actions |
| CI workflow | Added |
| Automatic Render deployment | Ready after deploy-hook secret is configured |
| Operational prediction logging | Added (does not log transaction identifiers or raw fields) |
| Drift detection / alerting | Not yet implemented |
| Central model registry / approval workflow | Not yet implemented |

This is a practical student mini-pipeline foundation. It should be described as **MLOps mini-pipeline implemented with remaining operational extensions**, not as enterprise-grade production MLOps until drift monitoring, alerting, and a managed model registry are configured.

## 8. Security and responsible-use note

This is an educational fraud-risk demonstration, not a payment authorization system. Do not enter real card numbers or personally identifiable information. The current form does not request card numbers. Protect the repository and deploy-hook secret, and review logs before sharing the public URL.
