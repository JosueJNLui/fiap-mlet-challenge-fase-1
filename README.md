# fiap-mlet-challenge-fase-1

[![Codecov](https://codecov.io/gh/JosueJNLui/fiap-mlet-challenge-fase-1/graph/badge.svg)](https://app.codecov.io/gh/JosueJNLui/fiap-mlet-challenge-fase-1)

Previsão de churn de clientes com base no dataset Telco Customer Churn. API REST em FastAPI servindo, por padrão, a **Logistic Regression (sklearn)** registrada no MLflow do DagsHub. O **MLP (PyTorch)** fica versionado como alternativa A/B-testável, selecionável via `MODEL_FLAVOR=pytorch` no `.env` (ver [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) §7.1).

## Arquitetura do sistema

API estruturada em camadas (DDD enxuto):

- **`src/api/`** (Interface): rotas FastAPI (`/health`, `/predict`), schemas Pydantic, dependências.
- **`src/application/`** (Casos de uso): `FeatureEngineer` (transformador `BaseEstimator/TransformerMixin`), `build_logreg_pipeline()` (sklearn `Pipeline` reprodutível), `ChurnPredictor` (modo Pipeline ou modo componentes), schemas pandera (`data_schemas.py`) e métricas de negócio.
- **`src/infrastructure/`** (Integrações externas): loader que busca o modelo registrado no MLflow do DagsHub. No flavor `sklearn` (default), carrega a `sklearn.Pipeline` empacotada (FeatureEngineer + StandardScaler + LogReg) como artefato único. No flavor `pytorch`, carrega o MLP e baixa `scaler.joblib` do mesmo run.
- **`src/main.py`** (Composição): `create_app()`, lifespan que carrega o modelo no startup (fail-fast), middleware de latência + logging JSON estruturado com `request_id` propagado.

Fluxo de uma predição:
1. Lifespan carrega o modelo (default: `Churn_LogReg_Final_Production` Pipeline empacotada; alternativo: `Churn_MLP_Final_Production` + scaler) na versão pinada → `app.state.predictor`.
2. Cliente faz `POST /predict` com payload Telco bruto (21 campos menos `customerID`).
3. Pydantic valida enums e ranges (422 em caso de erro).
4. `ChurnPredictor.predict()`:
   - **sklearn (default):** `pipeline.predict_proba(payload)` — a Pipeline interna executa FeatureEngineer → StandardScaler → LogReg; nenhum preprocessing manual no caminho de inferência.
   - **pytorch (alternativo):** `preprocess_one` → `scaler.transform` → tensor PyTorch + sigmoid.
   - Em ambos: comparação com threshold de negócio (default `0.2080` para LogReg; `0.20303` para o MLP, otimizados na mesma curva de lucro).
5. Resposta inclui `churn_probability`, `prediction`, `threshold`, `model_version`, `request_id`.

```mermaid
flowchart LR
    Client[Cliente]
    LB[Load Balancer]
    API[FastAPI + uvicorn]
    Pipe[sklearn.Pipeline<br/>FeatureEngineer → StandardScaler → LogReg]
    MLP[MLP PyTorch<br/>+ scaler joblib]
    Th[threshold de negócio]
    MLflow[(MLflow / DagsHub<br/>Model Registry)]

    Client -->|POST /predict| LB --> API
    API -->|flavor=sklearn| Pipe --> Th --> API
    API -.->|flavor=pytorch| MLP --> Th
    API -->|200 churn_probability| Client
    MLflow -.->|startup load| API
```

📚 **Documentação operacional:**
- [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) — performance, vieses, limitações, cenários de falha.
- [`docs/ARCHITECTURE_DEPLOY.md`](docs/ARCHITECTURE_DEPLOY.md) — decisão real-time, SLA, scaling, DR.
- [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) — arquitetura prática de deploy com Helm/Kubernetes e Terraform/AWS ECS.
- [`docs/MONITORING.md`](docs/MONITORING.md) — métricas técnicas/modelo/negócio, alertas, playbook.
- [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md) — fluxo TBD, Conventional Commits, SemVer.
- [`docs/CODE_GUIDELINES.md`](docs/CODE_GUIDELINES.md) — diretrizes de DDD, Clean Code e stack Python.

🧪 **Notebooks de pesquisa** (FIAP MLET Fase 1):
- `notebooks/eda.ipynb` cobre a **Etapa 1** (EDA + baselines DummyClassifier/Logistic Regression) e escreve no experimento MLflow `Churn-Predict-Telco-Etapa1-EDA`.
- `notebooks/modeling.ipynb` cobre a **Etapa 2** (MLP em PyTorch + ensembles, grid search, K-Fold, threshold otimizado, análise de trade-off FP×FN) e escreve em `Churn-Predict-Telco-Etapa2-Modelagem`.
- `notebooks/models-comparison.ipynb` consulta os **dois** experimentos para a comparação cruzada.
- Cálculos de lucro e custo de erro vivem em [`src/application/business_metrics.py`](src/application/business_metrics.py) — single source of truth compartilhada pelos três notebooks e pelo Model Card.

## Documentação interativa (Swagger / OpenAPI)

Com a API rodando, abra um dos endpoints abaixo no browser:

| URL | Descrição |
|---|---|
| <http://localhost:8000/docs> | Swagger UI — testar endpoints direto do browser (`Try it out`) |
| <http://localhost:8000/redoc> | ReDoc — documentação narrativa, ideal para leitura |
| <http://localhost:8000/openapi.json> | Spec OpenAPI 3.1 bruto, p/ gerar clientes (openapi-generator, etc.) |

Cada endpoint expõe `summary`, `description`, exemplos completos de payload e respostas, e modelos documentados para os erros `422` (validação) e `503` (modelo não carregado). Em produção, defina `DOCS_URL=` (vazio) no ambiente para desabilitar a UI sem alterar código.

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
  "prediction": true,
  "threshold": 0.2080,
  "model_version": "3",
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
| `MODEL_FLAVOR` | `sklearn` ou `pytorch` — define o caminho de inferência | `sklearn` |
| `MODEL_NAME` | Nome do modelo registrado | `Churn_LogReg_Final_Production` |
| `MODEL_VERSION` | Versão pinada (recomendado) | `3` |
| `PREDICTION_THRESHOLD` | Limiar de decisão | `0.2080` |
| `LOAD_MODEL_ON_STARTUP` | Se falso, pula carregamento (debug/dev) | `true` |

Sem credenciais válidas o startup falha por design (fail-fast com 401 do DagsHub).

### Modelo alternativo (A/B-testável): MLP

A API mantém dois caminhos de inferência selecionáveis sem deploy de código. Para servir o MLP em vez da LogReg, troque o bloco no `.env` para o "Fallback A/B-testável" descrito em [`.env.example`](.env.example) (`MODEL_FLAVOR=pytorch`, `MODEL_NAME=Churn_MLP_Final_Production`, `MODEL_VERSION=8`, `PREDICTION_THRESHOLD=0.20303030303030303`) e reinicie a API. Justificativa, equivalência estatística e cenários de uso estão em [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) §7.1.

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
make test             # suíte pytest hermética (não exige DagsHub)
make test-e2e-httpie  # E2E isolado com HTTPie contra servidor localhost
make test-cov         # testes + coverage.xml/htmlcov para Codecov
make lint             # ruff
make type-check       # ty
make check            # tudo + format check
```

Testes unitários/API usam `dependency_overrides` do FastAPI para injetar um
`FakePredictor`, então não precisam de credenciais nem rede.

O alvo `make test-e2e-httpie` executa `tests/e2e/test_httpie_api.py`: ele sobe
um `uvicorn` local com predictor determinístico e chama a API de fora do
processo usando HTTPie. A suíte cobre `/health`, `/predict` com payload válido,
border cases do payload Telco, erros comuns de parâmetros incorretos (`422`),
JSON malformado, método não permitido (`405`) e rota inexistente (`404`).

Para inspecionar status, headers e bodies de cada chamada:

```bash
E2E_HTTP_DEBUG=1 uv run pytest -s tests/e2e/test_httpie_api.py -q
```

### Cobertura com Codecov

A pipeline de CI executa `make test-cov` em todo `pull_request`, gera `coverage.xml` com `pytest-cov` e envia o relatório para o [Codecov](https://app.codecov.io/gh/JosueJNLui/fiap-mlet-challenge-fase-1) usando `codecov/codecov-action@v5`.

Para habilitar o upload no GitHub Actions, primeiro ative o repositório `JosueJNLui/fiap-mlet-challenge-fase-1` no [Codecov](https://app.codecov.io/gh/JosueJNLui/fiap-mlet-challenge-fase-1). Depois copie o token desse mesmo repositório e cadastre o segredo `CODECOV_TOKEN` em `Settings > Secrets and variables > Actions`.

Se o upload falhar com `Repository not found`, o relatório foi gerado, mas o Codecov não encontrou um repositório ativo para o token/slug usado. Revise se o repositório está habilitado no Codecov e se o `CODECOV_TOKEN` pertence exatamente a `JosueJNLui/fiap-mlet-challenge-fase-1`. O arquivo `codecov.yml` configura os status checks de projeto e patch com tolerância de 1% para pequenas variações de cobertura.

## Estrutura do repositório
```
├── data/
│   └── dataset/          # dataset original (Telco Customer Churn)
├── docs/                 # MODEL_CARD, ARCHITECTURE_DEPLOY, MONITORING, DEPLOYMENT, CONTRIBUTING, CODE_GUIDELINES
├── deploy/               # Helm/Kubernetes, Terraform/AWS ECS
├── notebooks/
│   ├── eda.ipynb               # Etapa 1: EDA + baselines (Dummy, LogReg) + MLflow
│   ├── modeling.ipynb          # Etapa 2: MLP PyTorch + ensembles + grid search + MLflow
│   └── models-comparison.ipynb # Etapas 1+2: comparação cross-experimento, trade-off, ranking
├── src/
│   ├── main.py                # create_app() + lifespan + middleware
│   ├── config.py              # Settings (pydantic-settings)
│   ├── api/                   # schemas, routes, dependencies
│   ├── application/           # preprocessing, transformers (FeatureEngineer),
│   │                          # pipeline (build_logreg_pipeline), data_schemas (pandera),
│   │                          # ChurnPredictor, business_metrics
│   └── infrastructure/        # mlflow_loader (DagsHub) — Pipeline (sklearn) ou modelo+scaler (pytorch)
├── tests/
│   ├── test_health_endpoint.py
│   ├── test_predict_endpoint.py
│   ├── application/      # preprocessing, predictor
│   └── integration/      # MLflow real (skipados por padrão)
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

Diretrizes detalhadas em [`docs/CODE_GUIDELINES.md`](docs/CODE_GUIDELINES.md) (DDD, Clean Code, Python).

## Mapa Etapas FIAP → artefatos do repo

| Etapa | Entrega | Onde está |
|---|---|---|
| **1.** EDA, qualidade, baselines (Dummy, LogReg), métrica técnica + de negócio, MLflow | Notebook de EDA + baselines registrados no MLflow | [`notebooks/eda.ipynb`](notebooks/eda.ipynb) (experimento `Churn-Predict-Telco-Etapa1-EDA`) |
| **2.** MLP em PyTorch + ensembles, comparação ≥4 métricas, trade-off FP×FN, MLflow | Tabela comparativa + MLP + artefatos | [`notebooks/modeling.ipynb`](notebooks/modeling.ipynb) e [`notebooks/models-comparison.ipynb`](notebooks/models-comparison.ipynb) (experimento `Churn-Predict-Telco-Etapa2-Modelagem`) |
| **3.** Refatoração modular, pipeline reprodutível (`sklearn.Pipeline` + `FeatureEngineer` custom), testes (pytest unitários + pandera schemas + smoke E2E), API FastAPI, logging + middleware, Makefile/ruff | Repositório refatorado + API funcional + testes | [`src/application/transformers.py`](src/application/transformers.py), [`src/application/pipeline.py`](src/application/pipeline.py), [`src/application/data_schemas.py`](src/application/data_schemas.py), [`tests/`](tests/), [`Makefile`](Makefile), [`pyproject.toml`](pyproject.toml) |
| **4.** Model Card, arquitetura de deploy, plano de monitoramento, README final | Documentação completa | [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md), [`docs/ARCHITECTURE_DEPLOY.md`](docs/ARCHITECTURE_DEPLOY.md), [`docs/MONITORING.md`](docs/MONITORING.md), [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md), este README |

## Conclusão

### Síntese dos relatos dos notebooks

**Etapa 1 — EDA + baselines** ([`notebooks/eda.ipynb`](notebooks/eda.ipynb)).
Dataset com 7.043 registros e 21 colunas, sem missings nominais (apenas 11 strings vazias em `TotalCharges` para clientes com `tenure=0`). Target desbalanceado em 73/27, com `tenure`, `MonthlyCharges` e `TotalCharges` confirmados como drivers de churn via Mann-Whitney U + Bonferroni. Multicolinearidade esperada entre `tenure` e `TotalCharges` (~0.83) tratada com `np.log1p`. A métrica de negócio adotada foi **lucro líquido** (LTV=R$500, custo de retenção=R$100), substituindo F1 pela assimetria de custos (FN custa 5× FP). Os baselines (`DummyClassifier` e `LogisticRegression(class_weight='balanced')`) já mostraram que a LogReg supera o piso por larga margem.

**Etapa 2 — MLP + ensembles** ([`notebooks/modeling.ipynb`](notebooks/modeling.ipynb)).
Comparação rigorosa entre LogReg, Random Forest, XGBoost e MLP (PyTorch) com K-Fold pareado, grid search e teste de Friedman+Nemenyi. A arquitetura do MLP foi mantida deliberadamente parsimoniosa: **uma única camada oculta** (`28 → hidden → BatchNorm → Dropout → 1`), `BCEWithLogitsLoss` com `pos_weight≈2.77` (equivalente ao `class_weight='balanced'`) e early stopping. Variantes com 2 camadas chegaram a ser exploradas e não trouxeram ganho mensurável — apenas mais parâmetros, mais tuning e maior risco de overfitting em ~5.6k amostras.

**Comparação cross-experimento** ([`notebooks/models-comparison.ipynb`](notebooks/models-comparison.ipynb)).
No hold-out, a Logistic Regression liderou simultaneamente as quatro frentes (recall 0.960, precisão 0.395, menor custo de FN R$7.500 e menor custo de FP R$54.900), atingindo lucro de **R$81.200**. O MLP campeão do grid ficou em **R$79.100** (Δ R$2.100), com Random Forest e XGBoost atrás em todas as métricas relevantes. Em CV, o MLP é estatisticamente **equivalente** à LogReg (Friedman+Nemenyi, p≈0.997), reforçando que o ganho não-linear esperado simplesmente não existe nesse volume e perfil de features.

### Escolha do modelo e a dificuldade de "vencer" a Logistic Regression

A decisão final foi promover a **Logistic Regression** como modelo de produção, sustentada por três pilares: (1) **financeiro** — maior lucro líquido absoluto; (2) **eficiência** — melhor controle de FP, evitando desperdício de verba de retenção; (3) **auditabilidade** — interpretabilidade direta dos coeficientes para as áreas de negócio. O MLP fica versionado como alternativa A/B-testável (`MODEL_FLAVOR=pytorch`) por ser estatisticamente equivalente — útil se o perfil dos dados mudar e justificar reavaliação, mas sem custo operacional adicional hoje.

A "dificuldade" em vencer a LogReg não é acidente: o sinal de churn no Telco é predominantemente **linear nas features pós-engenharia** (`Contract`, `tenure`, `InternetService`, `PaymentMethod` dominam), o `class_weight='balanced'` já calibra o desbalanceamento sem reamostragem, e a otimização de threshold por curva de lucro (não 0.5) extrai o ótimo operacional do modelo linear. Nesse regime, capacidade adicional vira variância, não viés removido — o que explica o resultado de Friedman e a opção pelo MLP **mais simples possível** (8 dims, 1 camada oculta) na fase de comparação.

### Conclusão geral do desafio

O desafio cobriu o ciclo end-to-end de um projeto de ML: formulação do problema com métrica de negócio explícita, EDA com validação estatística, baselines honestos, modelo "avançado" com PyTorch, refatoração para `sklearn.Pipeline` com transformador custom (`FeatureEngineer`), validação de schemas com pandera, API FastAPI com logging estruturado e middleware de latência, testes hermeticos + E2E com HTTPie, containerização e documentação operacional (Model Card, arquitetura de deploy, plano de monitoramento). O mapeamento Etapa→artefato fica explícito na tabela acima.

**Aprendizados com a stack:**
- **MLflow + DagsHub** funciona bem como Model Registry remoto sem infra própria, mas exige disciplina de pinagem (`MODEL_VERSION` fixo, alias `@production`) para evitar drift silencioso entre treino e serving.
- **`sklearn.Pipeline` empacotada** (FeatureEngineer → StandardScaler → LogReg) elimina toda a classe de bug de "preprocessing diferente em treino e inferência" — o caminho de servir vira `pipeline.predict_proba(payload)`, sem etapas manuais.
- **PyTorch para tabular** é viável e didático, mas em datasets pequenos com sinal linear o ROI sobre `sklearn` é baixo. A lição é validar empiricamente, não assumir que "rede neural ≥ baseline".
- **`uv` + `ruff` + `ty` + `pytest`** entregam um loop de desenvolvimento rápido: instalação reprodutível em segundos, lint+format unificados e type-check moderno sem o overhead histórico do mypy. Combinado com `Makefile` curto e Dockerfile com `mlflow-skinny` + `torch+cpu`, a imagem final cabe em ~1GB.
- **Lucro líquido como métrica primária** foi o que mais mudou as decisões — modelos com AUC competitivo (Random Forest, XGBoost) ficaram para trás em lucro, evidenciando que AUC e valor de negócio **não são intercambiáveis** e que a curva de threshold tem que ser otimizada explicitamente, não herdada do default 0.5.

## Sobre o projeto

| Item | Detalhe |
|------|---------|
| Curso | FIAP — Machine Learning Engineering |
| Fase | 1 |
| Dataset | [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
