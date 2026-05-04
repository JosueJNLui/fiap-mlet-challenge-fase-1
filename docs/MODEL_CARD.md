# Model Card — Churn Prediction (Telco Customer Churn)

> Documento padronizado seguindo o esquema Model Card (Mitchell et al., 2019) adaptado ao contexto do desafio FIAP MLET — Fase 1.

## 1. Detalhes do Modelo

| Campo | Valor |
|---|---|
| **Nome** | `Churn_LogReg_Final_Production` |
| **Versão em produção** | `2` (MLflow Model Registry — DagsHub), alias `@production` |
| **Algoritmo servido** | Logistic Regression (sklearn) — empacotado com `StandardScaler` (sklearn) e threshold de negócio |
| **Modelo alternativo (A/B-testável)** | MLP (PyTorch) — registrado como `Churn_MLP_Final_Production` v8; selecionável via `MODEL_FLAVOR=pytorch` no `.env` (ver §7.1) |
| **Data de treino** | Q1/2026 |
| **Owner / responsável** | Equipe FIAP MLET — Fase 1 |
| **Tracking** | DagsHub MLflow — `https://dagshub.com/JosueJNLui/fiap-mlet-challenge-fase-1.mlflow` |

### Hiperparâmetros da LogReg servida

- **Solver:** `lbfgs`.
- **Regularização:** L2 (default), `C=1.0`.
- **Tratamento de desbalanceamento:** `class_weight='balanced'`.
- **`max_iter`:** 1000.
- **`random_state`:** 42.

A escolha da LogReg como modelo servido decorre da equivalência estatística com o MLP no K-Fold pareado (Friedman + Nemenyi p≈0.997) somada à liderança em lucro de hold-out (R$ 81.200) e à vantagem operacional em **interpretabilidade, parsimônia e auditabilidade** (ver §5 e a célula de promoção em [`notebooks/models-comparison.ipynb`](../notebooks/models-comparison.ipynb)).

### Arquitetura do MLP (alternativa A/B-testável)

```
ChurnMLP(
  fc1: Linear(in_features=28, out_features=8)
  relu: ReLU
  bn1:  BatchNorm1d(8)
  drop: Dropout(p=0.15)
  fc2:  Linear(in_features=8, out_features=1)
)  # logit (BCEWithLogitsLoss)
```

- **Loss:** `BCEWithLogitsLoss` com `pos_weight` (classe positiva = churn).
- **Otimizador:** Adam, learning rate `1e-2`.
- **Batch size:** 128.
- **Early stopping:** patience = 20 epochs, max 100.
- **Hiperparâmetros vencedores** (grid search com 54 combinações): `hidden=8`, `lr=0.01`, `dropout=0.15`, `batch=128`.

---

## 2. Uso Pretendido

### Casos de uso primários
- Apoiar o time de retenção a priorizar clientes com alto risco de churn em campanhas proativas.
- Servir como score de risco em ferramentas internas (CRM, painéis de operação).

### Usuários-alvo
- Analistas de retenção e marketing.
- Sistemas internos via API REST (`POST /predict`).

### Usos **fora do escopo** (não recomendado)
- Decisões automatizadas sem revisão humana (negar serviço, alterar preços, encerrar contratos).
- Precificação dinâmica por cliente.
- Avaliação de risco de crédito ou fraude.
- Inferência sobre clientes B2B (o treino contempla apenas perfil residencial Telco).

---

## 3. Dados

### Dataset

- **Fonte:** [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
- **Volume:** 7.043 clientes × 21 colunas.
- **Target:** `Churn` (Yes/No) — desbalanceado: 73% No / 27% Yes.

### Qualidade

- 11 valores inválidos detectados em `TotalCharges` (strings em branco para `tenure=0`) → imputados como `MonthlyCharges × tenure`.
- Sem duplicatas.
- Sem missing values críticos após o tratamento de `TotalCharges`.

### Splits e validação

- Hold-out estratificado: 80% treino+validação / 20% teste.
- K-Fold estratificado (10 splits) sobre treino+validação para seleção de modelo.
- Avaliação final em hold-out de 20% (não tocado durante seleção).

### Tratamento de desbalanceamento

Comparação registrada em `notebooks/eda.ipynb`:

| Técnica | F1 (churn) | Decisão |
|---|---|---|
| `class_weight='balanced'` | **0.52** | ✅ **Selecionada** (estável, sem oversampling artificial) |
| SMOTE | 0.48 | Descartada |
| Random undersampling | 0.45 | Descartada |

---

## 4. Pré-processamento e Features

Pipeline reprodutível em `src/application/preprocessing.py` (espelha o notebook):

1. Encoding binário (Yes/No → 1/0) em 11 colunas.
2. `InternetService`: DSL/Fiber → 1, No → 0.
3. `log1p(TotalCharges)` (lida com `tenure=0` via NaN-safe).
4. **Feature engineering:**
   - `tenure_bucket` (binning: 0-12, 13-24, 25-48, 49+).
   - `avg_charges_per_month = TotalCharges / tenure`.
   - `charge_vs_expected = MonthlyCharges - avg_charges_per_month`.
   - `num_services` (soma dos 9 serviços contratados).
5. One-hot encoding manual: `gender`, `Contract`, `PaymentMethod`, `tenure_bucket`.
6. `reindex` para `EXPECTED_FEATURE_ORDER` (28 features fixas) — **single source of truth** entre treino e produção.
7. `StandardScaler` (sklearn) carregado como artefato MLflow do mesmo run do modelo.

---

## 5. Métricas de Performance

### Métricas técnicas (hold-out, 20% teste)

| Modelo | ROC-AUC | PR-AUC | F1 (churn) | Recall | Precision | Accuracy |
|---|---|---|---|---|---|---|
| DummyClassifier (maj.) | 0.500 | — | 0.000 | 0.00 | 0.00 | 0.73 |
| **Logistic Regression** | **0.849** | **0.670** | **0.560** | **0.96** | **0.40** | 0.74 |
| Random Forest (100, depth=10) | 0.841 | 0.645 | 0.551 | 0.95 | 0.40 | 0.74 |
| MLP (top-1 grid search) | 0.847 | 0.673 | 0.541 | 0.96 | 0.39 | 0.73 |

### Métrica de negócio: Lucro Líquido (BRL)

Definida em conjunto com a área:

```
Lucro = TP × LTV  −  FP × Custo_retencao  −  FN × LTV
        com LTV = R$ 500 e Custo_retencao = R$ 100
```

> **Auditabilidade:** o cálculo é centralizado em [`src/application/business_metrics.py`](../src/application/business_metrics.py) e consumido pelos três notebooks (`eda`, `modeling`, `models-comparison`). Qualquer ajuste de LTV, custo de retenção ou da fórmula passa por esse módulo + teste em [`tests/application/test_business_metrics.py`](../tests/application/test_business_metrics.py).

| Modelo | Lucro K-Fold | Lucro hold-out | FP × R$100 | FN × R$500 |
|---|---|---|---|---|
| DummyClassifier | — | **−R$ 187.000** | — | R$ 187.000 |
| **Logistic Regression** | R$ 33.120 | **R$ 81.200** | R$ 54.900 | R$ 7.500 |
| Random Forest | R$ 32.340 | R$ 77.800 | — | — |
| MLP (top-1) | R$ 33.120 | R$ 79.100 | R$ 55.200 | R$ 8.500 |

### Threshold de decisão (otimizado para negócio)

- **Threshold servido:** `0.2278` (LogReg, otimizado em validação cruzada 5-fold em `notebooks/eda.ipynb`).
- **Threshold do MLP alternativo:** `0.20303030303030303` (otimizado no K-Fold 10-fold em `notebooks/modeling.ipynb`).
- Cada threshold foi selecionado varrendo a curva PR do **seu próprio modelo** e maximizando o **lucro líquido** em validação (não o F1) — por isso são distintos.
- Reflete a assimetria de custo: cada FN custa 5× mais que cada FP (LTV R$ 500 vs custo retenção R$ 100), então ambos são deliberadamente baixos para favorecer recall.

### Comparação estatística (Friedman + Nemenyi, 10-fold pareado)

- **Friedman global:** p-value = `1.6e-04` → diferenças entre os 4+ modelos são significativas.
- **Post-hoc Nemenyi:** Logistic Regression ≈ MLP (top-1), p = `0.997` → **estatisticamente equivalentes**.
- **Decisão:** servir Logistic Regression (parsimônia, interpretabilidade), com MLP versionado como alternativa A/B-testável (ver §7.1).

> **Fonte dos números desta seção:** os resultados de Dummy/LogReg vêm das células finais de [`notebooks/eda.ipynb`](../notebooks/eda.ipynb); os de MLP/RandomForest/XGBoost de [`notebooks/modeling.ipynb`](../notebooks/modeling.ipynb) e [`notebooks/models-comparison.ipynb`](../notebooks/models-comparison.ipynb). Em caso de divergência, **os notebooks são autoritativos** — este Model Card replica os valores publicados.

---

## 6. Análise de Vieses e Subgrupos

Insights de `notebooks/eda.ipynb` (análise bivariada com Mann-Whitney p<0.05 nas variáveis numéricas):

| Subgrupo | Taxa de churn | Risco relativo | Comentário |
|---|---|---|---|
| `Contract = Month-to-month` | **42,7%** | 🔴 muito alto | Maior driver isolado de churn |
| `Contract = Two year` | 2,8% | 🟢 muito baixo | Cliente "âncora" |
| `InternetService = Fiber optic` | 41,9% | 🔴 alto | Possível insatisfação com produto |
| `InternetService = No` | 7,4% | 🟢 baixo | — |
| `PaymentMethod = Electronic check` | 45,3% | 🔴 muito alto | Forte sinal de menor engajamento |
| `PaperlessBilling = Yes` | 33,6% | 🟡 elevado | — |
| `tenure < 12 meses` | 47,4% | 🔴 muito alto | Janela crítica de onboarding |
| `tenure ≥ 49 meses` | 6,6% | 🟢 muito baixo | — |

### Vieses esperados / a monitorar

- **Concentração em Month-to-month + Electronic check + tenure baixo:** o modelo aprende fortemente esses padrões. Clientes que migrem para essas categorias serão imediatamente sinalizados — efeito desejado, mas pode causar **falsos positivos** em clientes recém-adquiridos via campanhas que naturalmente começam em mensal.
- **Gênero:** distribuição balanceada (Female ≈ Male) e Mann-Whitney não-significativo na análise → **não esperamos viés de gênero**, mas **não há análise formal de fairness por subgrupo** (gap conhecido — ver §8).
- **Idade (`SeniorCitizen`):** subgrupo é minoria (~16% do dataset). O modelo pode estar sub-otimizado para sêniores — recomenda-se monitorar Precision/Recall específico desse grupo.
- **Multicolinearidade:** `TotalCharges × tenure` correlação 0.826. Tratada com `log1p(TotalCharges)` + features derivadas, mas Logistic Regression mantém alguma redundância.

---

## 7. Cenários de Falha Esperados

| Cenário | Sintoma | Mitigação |
|---|---|---|
| **Drift de produto** (lançamento de novo plano) | Distribuição de `Contract`, `MonthlyCharges` fora do treino | Retreino; até lá, monitorar drift via KS test (ver `MONITORING.md`) |
| **Drift sazonal** (campanhas de Black Friday, fim de ano) | Pico de assinaturas Month-to-month e churn aumenta | Avaliar ROC-AUC mensal em amostra rotulada; reconfigurar threshold se necessário |
| **Mudança de público-alvo** (ex.: passar a vender B2B) | Modelo prediz churn alto consistentemente | Não usar — o modelo não foi treinado para B2B |
| **Falha no carregamento do scaler** | Predições com features não escaladas → scores degenerados | Lifespan fail-fast no startup; CI testa `load_predictor` (`tests/integration/`) |
| **Indisponibilidade do MLflow / DagsHub** | API falha ao subir | `MODEL_VERSION` pinado em env permite rollback determinístico; cache local opcional |
| **Necessidade de A/B test ou rollback do flavor** | Performance da LogReg degrada em segmento específico, ou queremos comparar com MLP | Trocar `MODEL_FLAVOR` / `MODEL_NAME` / `MODEL_VERSION` / `PREDICTION_THRESHOLD` no `.env` e reiniciar — sem deploy de código (ver §7.1) |
| **Payload mal-formado** | Erro 422 (Pydantic) | Validação automática antes de qualquer inferência |

### 7.1 Considerações operacionais — modelo servido vs alternativa

A API mantém **dois caminhos de inferência** (sklearn LogReg e PyTorch MLP) selecionáveis via `settings.model_flavor`. A LogReg é o default por parsimônia + equivalência estatística com o MLP (Nemenyi p≈0.997, ver §5). O MLP fica versionado como alternativa A/B-testável, útil em três cenários: (i) comparação prospectiva de performance em produção, (ii) rollback rápido se a LogReg degradar em algum segmento, (iii) análises pontuais que se beneficiem da capacidade não-linear.

**Receita de troca** (sem deploy de código): editar o `.env` para o bloco "Fallback A/B-testável: MLP (PyTorch)" descrito em [`.env.example`](../.env.example) e reiniciar a API. Os invariantes que **não** mudam: pipeline de pré-processamento, ordem das 28 features (`EXPECTED_FEATURE_ORDER`), e o caminho do scaler (`model_components/scaler.joblib`) — ambos os runs gravam o `StandardScaler` no mesmo path. O carregamento é dispatchado em [`src/infrastructure/mlflow_loader.py`](../src/infrastructure/mlflow_loader.py).

---

## 8. Limitações Conhecidas

- **Dataset estático (Kaggle).** Sem feed contínuo, não captura tendências reais de mercado.
- **Sem features temporais/comportamentais** (uso mensal, tickets de suporte, NPS).
- **Sem calibração formal de probabilidades.** O `churn_probability` retornado é o score do sigmoid — útil para ranking, mas não como probabilidade verdadeira sem reliability diagram.
- **Sem análise formal de fairness por subgrupo** (gênero, faixa etária, etnia — esta última ausente do dataset).
- **Threshold único global.** Não há thresholds segmentados por subgrupo, o que pode gerar desbalanço operacional (ex.: 90% do "high-risk pool" sendo Month-to-month).
- **Sem mecanismo de feedback loop** para o resultado real da retenção (cliente realmente saiu?), o que limita o monitoramento de negócio (ver `MONITORING.md`).

---

## 9. Considerações Éticas

- O dataset não contém atributos sensíveis explícitos (raça/etnia, religião, nacionalidade), porém `gender` está presente.
- O modelo é **support tool**, não decisor automático. Toda ação ofensiva (e.g., contato comercial, oferta) deve passar por revisão humana e regras de negócio.
- Logging estruturado preserva `request_id` mas não persiste payloads — alinhado com princípio de minimização de dados (LGPD/GDPR).

---

## 10. Como Reproduzir

```bash
# Instalar deps de desenvolvimento
make install-dev

# Reproduzir EDA e modelagem
jupyter lab notebooks/

# Servir o modelo localmente
export MLFLOW_TRACKING_USERNAME=<seu-user-dagshub>
export MLFLOW_TRACKING_PASSWORD=<seu-token-dagshub>
make run

# Testar
curl -X POST http://localhost:8000/predict -H 'Content-Type: application/json' -d @payload.json
```

Notebooks de referência (cada um escreve em um experimento MLflow distinto, listados em `models-comparison.ipynb`):

- [`notebooks/eda.ipynb`](../notebooks/eda.ipynb) — **Etapa 1**: EDA completa, baselines (DummyClassifier, Logistic Regression), métrica de negócio. Experimento: `Churn-Predict-Telco-Etapa1-EDA`.
- [`notebooks/modeling.ipynb`](../notebooks/modeling.ipynb) — **Etapa 2**: MLP em PyTorch (grid search 54 combinações, early stopping, K-Fold), ensembles (RandomForest, XGBoost), threshold otimizado, trade-off FP×FN. Experimento: `Churn-Predict-Telco-Etapa2-Modelagem`.
- [`notebooks/models-comparison.ipynb`](../notebooks/models-comparison.ipynb) — **Etapas 1+2**: leitura cruzada dos dois experimentos, ranking por lucro, análise de estabilidade validação vs. teste.

---

## 11. Documentos Relacionados

- [`ARCHITECTURE_DEPLOY.md`](ARCHITECTURE_DEPLOY.md) — arquitetura de deploy (real-time vs batch, SLA, scaling).
- [`MONITORING.md`](MONITORING.md) — plano de monitoramento (técnico, modelo, negócio, alertas, playbook).
- [`../README.md`](../README.md) — instruções de setup e uso da API.
