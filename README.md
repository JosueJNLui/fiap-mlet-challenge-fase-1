# fiap-mlet-challenge-fase-1

Previsão de churn de clientes com base no dataset Telco Customer Churn.

## Arquitetura do sistema

O projeto é uma API REST simples construída com **FastAPI** e estruturada para separar responsabilidades entre:

- **Interface/API:** `src/main.py` expõe o endpoint `/health` e registra métricas de latência.
- **Observabilidade:** middleware que adiciona `X-Process-Time` em todas as respostas e produz logs estruturados em JSON.
- **Dependências:** gerenciadas via `pyproject.toml` e instaladas com `uv`.

Fluxo principal:
1. O cliente faz uma requisição HTTP.
2. O middleware mede latência e registra um log JSON.
3. O endpoint processa a requisição e retorna o resultado.

### Componentes

- `src/main.py`: aplicação FastAPI, middleware e endpoint de saúde
- `tests/test_health_endpoint.py`: teste de integração do endpoint `/health`
- `Dockerfile`: imagem Docker baseada em Python 3.13-slim com `uv` instalado
- `data/dataset/`: dataset original
- `data/mlflow/`: repositório de experimentos MLflow
- `models/`: espaço para modelos treinados

## Estrutura do repositório
```
├── data/
│   ├── dataset/          # dataset original
│   └── mlflow/           # banco de experimentos (não versionado)
├── docs/                 # documentação adicional
├── models/               # modelos treinados (não versionados)
├── notebooks/            # EDA e experimentação
├── src/
│   ├── __init__.py
│   ├── INSTRUCTION.md    # diretrizes de arquitetura e qualidade
│   └── main.py           # script final de produção
├── tests/
│   └── test_health_endpoint.py
├── pyproject.toml
└── uv.lock
```

## Como construir usando Docker

1. Construa a imagem:
```bash
docker build -t fiap-mlet-challenge-fase-1:latest .
```
2. Execute o container:
```bash
docker run --rm -p 8000:8000 fiap-mlet-challenge-fase-1:latest
```
3. Verifique o endpoint de saúde:
```bash
curl http://localhost:8000/health
```

## Como testar

1. Instale as dependências de desenvolvimento:
```bash
uv sync --extra dev
```
2. Execute os testes:
```bash
uv run python -m pytest
```

O projeto atualmente inclui um teste de endpoint de saúde em `tests/test_health_endpoint.py`.

## Como executar localmente

1. Instale as dependências de produção:
```bash
uv sync
```
2. Execute a aplicação com Uvicorn:
```bash
uv run uvicorn src.main:app --host 0.0.0.0 --port 8000
```

3. Abra o navegador ou use `curl`:
```bash
curl http://localhost:8000/health
```

> Alternativa: `uv run src/main.py` também pode ser usada para iniciar o serviço diretamente.

## Instruções usadas

O projeto segue as diretrizes definidas em `src/INSTRUCTION.md`, incluindo:

- uso de **Python 3.13+** e **FastAPI**
- dependências gerenciadas via `pyproject.toml` e `uv`
- separação de responsabilidades no estilo **DDD / Clean Code**
- testes escritos com **pytest**
- logs estruturados e middleware de latência
- validação de tipos e estilo com **ty** e **ruff**

## Sobre o projeto

| Item | Detalhe |
|------|---------|
| Curso | FIAP — Machine Learning Engineering |
| Fase | 1 |
| Dataset | [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |

## Integrantes

| Nome | GitHub |
|------|--------|
|      |        |
