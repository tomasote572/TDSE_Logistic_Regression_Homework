## Heart Disease Risk Prediction — Logistic Regression
Author: Tomas Felipe Ramirez Alvarez

### Summary
This project implements logistic regression from scratch (NumPy) to predict the presence of heart disease. The notebook includes full EDA with descriptive statistics, missing values analysis, feature distributions, correlation heatmap, and per-class boxplots. It also covers feature selection and normalization, implementation of core functions (sigmoid, cost, gradient descent), training and evaluation with multiple metrics, decision-boundary visualizations for three feature pairs, an extended L2 regularization sweep, and Docker-based FastAPI deployment for real-time inference.

### Dataset
- Source: Kaggle — neurocipher/heartdisease (https://www.kaggle.com/datasets/neurocipher/heartdisease)
- File included: `Heart_Disease_Prediction.csv`
- Samples: 303
- Target: `Heart Disease` (values: `Presence`, `Absence`) — the notebook binarizes it to 1/0.

### Structure and deliverables
- `cuaderno1.ipynb`: Notebook with the full workflow (EDA, training, visualizations, regularization sweep, and model saving).
- `Heart_Disease_Prediction.csv`: Dataset (should be present locally).
- `best_model.npz`: File produced after the lambda sweep (contains `w`, `b`, `mu`, `sigma`, `features`).
- `app.py`: FastAPI REST service with `/predict` (single) and `/predict_batch` (batch) endpoints.
- `Dockerfile.api`: Docker image definition — builds and runs the FastAPI service.
- `requirements-api.txt`: Python dependencies for the API.

### Requirements
- Python 3.8+
- Packages: `pandas`, `numpy`, `matplotlib`

Quick install:

```powershell
pip install pandas numpy matplotlib
```


### Deployment — Docker (quick start)

**Prerequisites:** Docker installed and running.

**1. Build the image**

```bash
docker build -f Dockerfile.api -t heart-disease-api .
```

**2. Run the container**

```bash
docker run -p 8000:8000 heart-disease-api
```

The API will be available at `http://localhost:8000`.

**3. Interactive documentation (Swagger UI)**

Open `http://localhost:8000/docs` in a browser to explore and test all endpoints interactively.

**4. Test the API**

Single prediction (`POST /predict`):

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"Age": 80, "Cholesterol": 400, "RestingBP": 180, "MaxHR": 90, "Oldpeak": 4.0, "CA": 3}}'
```

Batch prediction (`POST /predict_batch`):

```bash
curl -X POST http://localhost:8000/predict_batch \
  -H "Content-Type: application/json" \
  -d '{"instances": [{"Age":63,"Cholesterol":233,"RestingBP":145,"MaxHR":150,"Oldpeak":2.3,"CA":0}, {"Age":37,"Cholesterol":250,"RestingBP":130,"MaxHR":170,"Oldpeak":1.4,"CA":1}]}'
```

**5. Stop the container**

```bash
# Find the container ID
docker ps
# Stop it
docker stop <container_id>
```


### evidence

- ![text](images/1.png) 
- ![text](images/2.png) 
- ![text](images/3.png) 
- ![text](images/4.png)
- ![alt text](images/5.png) 
- ![alt text](images/6.png) 
- ![alt text](images/7.png)

### Recent results (examples)

- Single example sent to `/predict`:
   - Input: `{"features": {"Age": 80, "Cholesterol": 400, "RestingBP": 180, "MaxHR": 90, "Oldpeak": 4.0, "CA": 3}}`
   - Result: `{"probability": 0.99..., "prediction": 1}` — the model estimates a high risk (positive prediction) for this extreme case.

- Example sent to `/predict_batch` (two instances with moderate values):
   - Input: two records with Age 63 and 37 (cholesterol, blood pressure, etc.).
   - Result: probabilities ≈ 0.47 and 0.47 → `prediction: 0` (absence under the 0.5 threshold).

**Short interpretation**
- The default decision threshold is 0.5; values above classify as `1` (presence) and below as `0` (absence).
- A probability close to 0.5 indicates uncertainty; clinical decisions should not be based solely on this output.
- Before production: evaluate performance on a held-out test set (accuracy, precision, recall, F1, AUC), check calibration, and log predictions to monitor data drift.

### Conclusions
1. Training logistic regression from scratch provides a clear baseline and interpretable coefficients; performance metrics are computed and shown in the notebook.
2. L2 regularization reduces weight magnitude and can improve generalization; tuning lambda demonstrated a trade-off between bias and variance.
3. Some feature pairs (such as Age vs. Cholesterol) exhibit better class separability; feature engineering or non-linear models could further improve results.
4. Saving model parameters (`best_model.npz`) simplifies deployment workflows (SageMaker or lightweight REST services) for real-time risk scoring.
 

### Note: Since deployment to AWS SageMaker failed, the model was packaged in Docker with a FastAPI service, exposing `/predict` and `/predict_batch` endpoints for dynamic real-time inference. Follow the Docker quick-start section above to run it locally.
