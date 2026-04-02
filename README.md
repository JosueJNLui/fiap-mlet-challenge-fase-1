# fiap-mlet-challenge-fase-1

Previsão de churn de clientes com base no dataset Telco Customer Churn.

## Sobre o projeto

| Item | Detalhe |
|------|---------|
| Curso | FIAP — Machine Learning Engineering |
| Fase | 1 |
| Dataset | [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |

## Estrutura do repositório
```
├── data/
│   ├── dataset/          # dataset original
│   └── mlflow/           # banco de experimentos (não versionado)
├── docs/                 # documentação adicional
├── models/               # modelos treinados (não versionados)
├── notebooks/            # EDA e experimentação
├── src/
│   └── main.py           # script final de produção
├── tests/
├── pyproject.toml
└── uv.lock
```

## Setup

### 1. Instalar o uv

Siga as instruções oficiais: https://docs.astral.sh/uv/getting-started/installation/

### 2. EDA e notebooks
```bash
uv sync --extra dev      # instala dependências de desenvolvimento
uv run jupyter lab       # abre o Jupyter
```

### 3. Script final
```bash
uv sync                  # instala apenas dependências de produção
uv run src/main.py       # executa o script
```

## ML Canvas

- [Rascunho no Excalidraw](https://excalidraw.com/#json=brwKEsokY4bOocmWOZ99T,_CNgHlfQxs-9-s0miVXkWw)

## Integrantes

| Nome | GitHub |
|------|--------|
|      |        |
