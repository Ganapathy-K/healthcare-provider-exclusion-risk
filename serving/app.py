# 4.1 Imports and app initialisation
from fastapi import FastAPI
from pydantic import BaseModel
import xgboost as xgb
import pandas as pd
import json
from pathlib import Path

app = FastAPI(title="Healthcare Provider Exclusion Risk API")

# 4.2 Load model at startup
MODEL_PATH = Path(__file__).parent / "model.ubj"
model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)

# 4.3 Load target encoding maps
ENCODING_MAPS_PATH = Path(__file__).parent / "encoding_maps.json"
with open(ENCODING_MAPS_PATH) as file:
    encoding_maps = json.load(file)

# 4.4 Feature column order — must match training exactly
FEATURE_COLUMNS = [
    "Entity Type Code",
    "Provider Business Mailing Address State Name",
    "Provider Business Mailing Address Telephone Number",
    "Provider Business Practice Location Address State Name",
    "Healthcare Provider Taxonomy Code_1",
    "Provider License Number State Code_1",
    "Provider Enumeration Year",
    "Last Update Year",
    "Provider Sex Code_F",
    "Provider Sex Code_M",
    "Provider Sex Code_U",
    "Healthcare Provider Primary Taxonomy Switch_1_N",
    "Healthcare Provider Primary Taxonomy Switch_1_Y",
    "Is Sole Proprietor_N",
    "Is Sole Proprietor_X",
    "Is Sole Proprietor_Y",
]

# 4.5 Request schema — categorical fields accepted as raw strings
class ProviderInput(BaseModel):
    Entity_Type_Code: int
    Provider_Business_Mailing_Address_State_Name: str
    Provider_Business_Mailing_Address_Telephone_Number: float
    Provider_Business_Practice_Location_Address_State_Name: str
    Healthcare_Provider_Taxonomy_Code_1: str
    Provider_License_Number_State_Code_1: str
    Provider_Enumeration_Year: int
    Last_Update_Year: int
    Provider_Sex_Code: str                              # F, M, or U
    Healthcare_Provider_Primary_Taxonomy_Switch_1: str  # N or Y
    Is_Sole_Proprietor: str                             # N, X, or Y

# 4.6 Encode raw input into the 16 model features
def encode_input(data: ProviderInput) -> pd.DataFrame:
    taxonomy_mean = encoding_maps["Healthcare Provider Taxonomy Code_1"].get(data.Healthcare_Provider_Taxonomy_Code_1, 0.0)
    mailing_state_mean = encoding_maps["Provider Business Mailing Address State Name"].get(data.Provider_Business_Mailing_Address_State_Name, 0.0)
    practice_state_mean = encoding_maps["Provider Business Practice Location Address State Name"].get(data.Provider_Business_Practice_Location_Address_State_Name, 0.0)
    license_state_mean = encoding_maps["Provider License Number State Code_1"].get(data.Provider_License_Number_State_Code_1, 0.0)

    row = {
        "Entity Type Code": data.Entity_Type_Code,
        "Provider Business Mailing Address State Name": mailing_state_mean,
        "Provider Business Mailing Address Telephone Number": data.Provider_Business_Mailing_Address_Telephone_Number,
        "Provider Business Practice Location Address State Name": practice_state_mean,
        "Healthcare Provider Taxonomy Code_1": taxonomy_mean,
        "Provider License Number State Code_1": license_state_mean,
        "Provider Enumeration Year": data.Provider_Enumeration_Year,
        "Last Update Year": data.Last_Update_Year,
        "Provider Sex Code_F": int(data.Provider_Sex_Code == "F"),
        "Provider Sex Code_M": int(data.Provider_Sex_Code == "M"),
        "Provider Sex Code_U": int(data.Provider_Sex_Code == "U"),
        "Healthcare Provider Primary Taxonomy Switch_1_N": int(data.Healthcare_Provider_Primary_Taxonomy_Switch_1 == "N"),
        "Healthcare Provider Primary Taxonomy Switch_1_Y": int(data.Healthcare_Provider_Primary_Taxonomy_Switch_1 == "Y"),
        "Is Sole Proprietor_N": int(data.Is_Sole_Proprietor == "N"),
        "Is Sole Proprietor_X": int(data.Is_Sole_Proprietor == "X"),
        "Is Sole Proprietor_Y": int(data.Is_Sole_Proprietor == "Y"),
    }
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)

# 4.7 Health check endpoint
@app.get("/health")
def health():
    return {"status": "ok"}

# 4.8 Prediction endpoint
@app.post("/predict")
def predict(provider: ProviderInput):
    input_dataframe = encode_input(provider)
    exclusion_probability = float(model.predict_proba(input_dataframe)[0][1])
    prediction = int(model.predict(input_dataframe)[0])
    return {
        "excluded": bool(prediction),
        "exclusion_probability": round(exclusion_probability, 4),
        "risk_tier": "high" if exclusion_probability >= 0.5 else "low",
    }
