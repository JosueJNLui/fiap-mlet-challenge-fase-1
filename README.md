# fiap-mlet-challenge-fase-1

Previsão de churn de clientes com base no dataset Telco Customer Churn. API REST em FastAPI servindo o modelo MLP final (PyTorch) registrado no MLflow do DagsHub.

## Arquitetura do sistema

API estruturada em camadas (DDD enxuto):

- **`src/api/`** (Interface): rotas FastAPI (`/health`, `/predict`), schemas Pydantic, dependências.
- **`src/application/`** (Casos de uso): pré-processamento de features (replica do notebook) e `ChurnPredictor` (modelo + scaler + threshold).
- **`src/infrastructure/`** (Integrações externas): loader que busca o modelo registrado no MLflow do DagsHub e baixa o artefato `scaler.joblib` do mesmo run.
- **`src/main.py`** (Composição): `create_app()`, lifespan que carrega o modelo no startup (fail-fast), middleware de latência + logging JSON estruturado com `request_id` propagado.

Fluxo de uma predição:
1. Lifespan carrega `Churn_MLP_Final_Production` (versão pinada) e `scaler.joblib` do MLflow → `app.state.predictor`.
2. Cliente faz `POST /predict` com payload Telco bruto (21 campos menos `customerID`).
3. Pydantic valida enums e ranges (422 em caso de erro).
4. `ChurnPredictor.predict()`: preprocessing → scaler → tensor PyTorch → sigmoid → threshold (0.20303, otimizado em curva PR de negócio).
5. Resposta inclui `churn_probability`, `prediction`, `threshold`, `model_version`, `request_id`.

## Endpoints

### `GET /health`
Retorna `{"status":"ok","timestamp":"...Z"}` com headers `X-Process-Time` e `X-Request-ID`.

### `POST /predict`

Recebe um payload Telco bruto (21 campos do dataset, menos `customerID`) e retorna a probabilidade de churn aplicando o pré-processamento + scaler + threshold de negócio.

Exemplo com `curl`:

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
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
    "TotalCharges": 1850.0
  }'
```

Resposta (200):
```json
{
  "churn_probability": 0.42,
  "prediction": false,
  "threshold": 0.20303030303030303,
  "model_version": "8",
  "request_id": "9f4a..."
}
```

Para rastrear uma chamada específica nos logs, envie um `X-Request-ID` próprio — ele é ecoado no body e nos headers da resposta:

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -H 'X-Request-ID: my-trace-id-123' \
  -d @payload.json
```

Erros de validação retornam `422` (ex.: `Contract` fora dos enums permitidos, `tenure` negativo).

## Configuração

A API lê variáveis de ambiente (ou `.env` local). Há um template versionado no repo — basta copiar e preencher:

```bash
cp .env.example .env
```

Depois edite `.env` e configure suas credenciais do DagsHub:

- `MLFLOW_TRACKING_USERNAME`: seu usuário do DagsHub (o token só autentica como o dono dele — não use o usuário de outra pessoa).
- `MLFLOW_TRACKING_PASSWORD`: seu access token, gerado em <https://dagshub.com/user/settings/tokens>.

O arquivo `.env` está no `.gitignore`, então o token não será commitado.

Variáveis disponíveis:

| Variável | Descrição | Default |
|---|---|---|
| `MLFLOW_TRACKING_USERNAME` | Seu usuário DagsHub | — (obrigatório) |
| `MLFLOW_TRACKING_PASSWORD` | Seu token DagsHub | — (obrigatório) |
| `MLFLOW_TRACKING_URI` | URI do MLflow no DagsHub | `https://dagshub.com/JosueJNLui/fiap-mlet-challenge-fase-1.mlflow` |
| `MODEL_NAME` | Nome do modelo registrado | `Churn_MLP_Final_Production` |
| `MODEL_VERSION` | Versão pinada (recomendado) | `8` |
| `PREDICTION_THRESHOLD` | Limiar de decisão | `0.20303030303030303` |
| `LOAD_MODEL_ON_STARTUP` | Se falso, pula carregamento (debug/dev) | `true` |

Sem credenciais válidas o startup falha por design (fail-fast com 401 do DagsHub).

## Como executar

### Local (com `uv`)
```bash
make install-dev
export MLFLOW_TRACKING_PASSWORD=<seu-token-dagshub>
make run
# em outro terminal
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -H 'Content-Type: application/json' -d @payload.json
```

### Docker
```bash
make docker-build
docker run --rm -e MLFLOW_TRACKING_PASSWORD=$TOKEN -p 8000:8000 fiap-mlet-challenge-fase-1:latest
```

A imagem usa `mlflow-skinny` + `torch+cpu`, pesa ~1GB (vs ~3.5GB com defaults).

## Como testar

```bash
make test          # 14 testes hermético (não exigem DagsHub)
make lint          # ruff
make type-check    # ty
make check         # tudo + format check
```

Testes usam `dependency_overrides` do FastAPI para injetar um `FakePredictor`, então não precisam de credenciais nem rede.

## Estrutura do repositório
```
├── data/
│   ├── dataset/          # dataset original
│   └── mlflow/           # store local (não usado em prod)
├── docs/
├── models/
├── notebooks/            # EDA, treinamento, comparações
├── src/
│   ├── main.py           # create_app() + lifespan + middleware
│   ├── config.py         # Settings (pydantic-settings)
│   ├── api/              # schemas, routes, dependencies
│   ├── application/      # preprocessing, ChurnPredictor
│   └── infrastructure/   # mlflow_loader (DagsHub)
├── tests/
│   ├── test_health_endpoint.py
│   ├── test_predict_endpoint.py
│   └── application/      # preprocessing, predictor
├── Dockerfile
├── Makefile
├── pyproject.toml
└── uv.lock
```

## Stack

- Python 3.13, FastAPI, Pydantic v2, pydantic-settings
- PyTorch (CPU), scikit-learn, pandas, numpy, joblib
- MLflow-skinny client (DagsHub remoto)
- `uv` para deps, `ruff` lint+format, `ty` type-check, `pytest` testes
- Docker (`python:3.13-slim` + `uv`)

Diretrizes detalhadas em `src/INSTRUCTION.md` (DDD, Clean Code, Python).

## Sobre o projeto

| Item | Detalhe |
|------|---------|
| Curso | FIAP — Machine Learning Engineering |
| Fase | 1 |
| Dataset | [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
