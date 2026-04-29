from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

YesNo = Literal["Yes", "No"]
YesNoNoInternet = Literal["Yes", "No", "No internet service"]
YesNoNoPhone = Literal["Yes", "No", "No phone service"]


class HealthResponse(BaseModel):
    """Liveness/readiness probe payload."""

    status: Literal["ok"] = Field(
        default="ok",
        description="Sempre `ok` quando a API está pronta para receber tráfego.",
    )
    timestamp: str = Field(
        description="Horário UTC do health-check, em ISO-8601 com sufixo `Z`.",
        examples=["2026-04-29T19:34:00Z"],
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "ok",
                "timestamp": "2026-04-29T19:34:00Z",
            }
        }
    )


class PredictRequest(BaseModel):
    """Payload bruto do Telco Customer Churn (um cliente).

    Espelha as colunas de `data/dataset/telco_customer_churn.csv` exceto
    `customerID` (descartada no treino). Os enums e ranges replicam o dataset
    original — assim o consumidor não precisa conhecer o pré-processamento
    interno (encoding, scaler, threshold).
    """

    gender: Literal["Male", "Female"] = Field(
        description="Gênero do cliente (`Male` ou `Female`).",
    )
    SeniorCitizen: Literal[0, 1] = Field(
        description="`1` se for idoso (≥65), `0` caso contrário.",
    )
    Partner: YesNo = Field(description="Possui cônjuge/parceiro?")
    Dependents: YesNo = Field(description="Possui dependentes (filhos, etc.)?")
    tenure: int = Field(
        ge=0,
        le=120,
        description="Meses como cliente. `0` significa contratação no mês corrente.",
    )
    PhoneService: YesNo = Field(description="Possui linha de telefone fixa.")
    MultipleLines: YesNoNoPhone = Field(description="Múltiplas linhas telefônicas.")
    InternetService: Literal["DSL", "Fiber optic", "No"] = Field(
        description="Tipo de conexão à internet contratada.",
    )
    OnlineSecurity: YesNoNoInternet = Field(description="Add-on de segurança online.")
    OnlineBackup: YesNoNoInternet = Field(description="Add-on de backup em nuvem.")
    DeviceProtection: YesNoNoInternet = Field(
        description="Add-on de proteção de equipamento."
    )
    TechSupport: YesNoNoInternet = Field(description="Suporte técnico premium.")
    StreamingTV: YesNoNoInternet = Field(description="Add-on de streaming de TV.")
    StreamingMovies: YesNoNoInternet = Field(
        description="Add-on de streaming de filmes."
    )
    Contract: Literal["Month-to-month", "One year", "Two year"] = Field(
        description="Tipo de contrato. Mensal tende a maior churn.",
    )
    PaperlessBilling: YesNo = Field(description="Recebe fatura digital.")
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ] = Field(description="Forma de pagamento mensal.")
    MonthlyCharges: float = Field(
        ge=0,
        description="Valor da fatura mensal vigente, em USD.",
    )
    # The raw CSV stores TotalCharges as a string with occasional " " values
    # for tenure==0; accept both float and str and coerce server-side.
    TotalCharges: float | str = Field(
        description=(
            "Valor total cobrado até hoje, em USD. Aceita `float` ou string "
            '(ex.: `" "` para clientes com `tenure=0`, conforme o CSV original).'
        ),
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 24,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "DSL",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "Yes",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "One year",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 75.5,
                "TotalCharges": 1850.0,
            }
        }
    )


class PredictResponse(BaseModel):
    """Saída da predição. `prediction=True` significa que o modelo espera churn."""

    churn_probability: float = Field(
        ge=0,
        le=1,
        description="Probabilidade calibrada de churn no intervalo `[0, 1]`.",
        examples=[0.42],
    )
    prediction: bool = Field(
        description=(
            "`true` quando `churn_probability >= threshold`. Use este campo "
            "como decisão final — o threshold já incorpora a regra de negócio."
        ),
    )
    threshold: float = Field(
        description=(
            "Limiar de decisão otimizado em curva PR para minimizar custo "
            "de churn evitado. Devolvido para auditoria."
        ),
        examples=[0.20303030303030303],
    )
    model_version: str = Field(
        description="Versão do modelo registrada no MLflow que serviu a resposta.",
        examples=["8"],
    )
    request_id: str = Field(
        description=(
            "Eco do header `X-Request-ID`, ou um UUID gerado pela API. "
            "Use este valor para cruzar logs e métricas."
        ),
        examples=["9f4a3f7b-2e1c-4b8a-9c5d-1a2b3c4d5e6f"],
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "churn_probability": 0.42,
                "prediction": True,
                "threshold": 0.20303030303030303,
                "model_version": "8",
                "request_id": "9f4a3f7b-2e1c-4b8a-9c5d-1a2b3c4d5e6f",
            }
        }
    )


class ValidationErrorItem(BaseModel):
    """Detalhe individual emitido pelo Pydantic em respostas 422."""

    loc: list[str | int] = Field(
        description="Caminho do campo inválido (ex.: `['body', 'tenure']`).",
        examples=[["body", "tenure"]],
    )
    msg: str = Field(
        description="Mensagem legível do erro de validação.",
        examples=["Input should be greater than or equal to 0"],
    )
    type: str = Field(
        description="Identificador do tipo de erro do Pydantic.",
        examples=["greater_than_equal"],
    )


class ValidationErrorResponse(BaseModel):
    """Resposta padrão do FastAPI para payloads inválidos (HTTP 422)."""

    detail: list[ValidationErrorItem]

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "detail": [
                    {
                        "loc": ["body", "tenure"],
                        "msg": "Input should be greater than or equal to 0",
                        "type": "greater_than_equal",
                    }
                ]
            }
        }
    )


class ServiceUnavailableResponse(BaseModel):
    """Modelo carregado com falha — API não está pronta para servir predições."""

    detail: str = Field(
        description="Mensagem explicando o motivo da indisponibilidade.",
        examples=["Predictor not loaded"],
    )
