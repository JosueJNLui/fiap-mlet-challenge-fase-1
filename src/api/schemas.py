from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

YesNo = Literal["Yes", "No"]
YesNoNoInternet = Literal["Yes", "No", "No internet service"]
YesNoNoPhone = Literal["Yes", "No", "No phone service"]


class HealthResponse(BaseModel):
    status: str = "ok"
    timestamp: str


class PredictRequest(BaseModel):
    """Raw Telco Churn payload (one customer).

    Mirrors the columns of `data/dataset/telco_customer_churn.csv` minus
    `customerID` (which the training pipeline drops). Field types and enum
    values match the dataset exactly so consumers don't have to know about
    internal preprocessing.
    """

    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=120)
    PhoneService: YesNo
    MultipleLines: YesNoNoPhone
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: YesNoNoInternet
    OnlineBackup: YesNoNoInternet
    DeviceProtection: YesNoNoInternet
    TechSupport: YesNoNoInternet
    StreamingTV: YesNoNoInternet
    StreamingMovies: YesNoNoInternet
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0)
    # The raw CSV stores TotalCharges as a string with occasional " " values
    # for tenure==0; accept both float and str and coerce server-side.
    TotalCharges: float | str


class PredictResponse(BaseModel):
    """Prediction output. `prediction=True` means the model expects churn."""

    churn_probability: float = Field(ge=0, le=1)
    prediction: bool
    threshold: float
    model_version: str
    request_id: str
