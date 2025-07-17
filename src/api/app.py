from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd

# Absolute model path
MODEL_PATH = r"C:\Users\teamp\Desktop\Cybersecurity Intrusion Detection\src\models\output\IDS.joblib"

app = FastAPI(
    title="Advanced Intrusion Detection System API",
    description="API for network intrusion detection prediction",
    version="1.0.0"
)

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model at startup: {e}")

class NetworkData(BaseModel):
    duration: float = Field(..., description="Duration of the connection")
    src_bytes: float = Field(..., description="Bytes sent from source")
    dst_bytes: float = Field(..., description="Bytes sent to destination")
    wrong_fragment: float = Field(..., description="Number of wrong fragments")
    urgent: float = Field(..., description="Number of urgent packets")
    # Add all other features here with exact names and types

@app.post("/predict")
async def predict(data: NetworkData):
    try:
        input_df = pd.DataFrame([data.dict()])
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0].tolist() if hasattr(model, "predict_proba") else None

        return {
            "prediction": int(prediction),
            "probability": probability
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
