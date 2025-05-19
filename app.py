from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Modeli yükle
model = joblib.load('xgb_high_model.pkl')

# Modelin beklediği feature isimleri (eğitimde kullandığın ile aynı)
FEATURE_NAMES = ['Open', 'Low', 'Close', 'Adj Close', 'Volume',
                 'High_lag_1', 'High_lag_2', 'High_lag_3', 'High_lag_4', 'High_lag_5',
                 'High_ma_3', 'High_ma_7', 'weekday']

# API'ye gönderilecek JSON veri için Pydantic model tanımı
class Features(BaseModel):
    Open: float
    Low: float
    Close: float
    Adj_Close: float
    Volume: float
    High_lag_1: float
    High_lag_2: float
    High_lag_3: float
    High_lag_4: float
    High_lag_5: float
    High_ma_3: float
    High_ma_7: float
    weekday: int

@app.post("/predict")
def predict(features: Features):
    # JSON'dan DataFrame oluştur
    data_dict = features.dict()
    # API'ye gelen "Adj_Close" ile modeldeki "Adj Close" farkına dikkat, onu düzeltelim
    data_dict['Adj Close'] = data_dict.pop('Adj_Close')
    df = pd.DataFrame([data_dict], columns=FEATURE_NAMES)

    # Tahmin yap
    prediction = model.predict(df)[0]
    prediction_float = float(prediction)  # numpy tipini float'a çevir
    return {"predicted_high": prediction_float}
