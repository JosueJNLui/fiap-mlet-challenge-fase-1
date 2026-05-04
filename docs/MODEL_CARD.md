# Model Card: Churn Prediction (Telco Customer Churn)

> Documento padronizado seguindo o esquema Model Card (Mitchell et al., 2019) adaptado ao contexto do desafio FIAP MLET, Fase 1.

## 1. Detalhes do Modelo

| Campo | Valor |
|---|---|
| **Nome** | `Churn_LogReg_Final_Production` |
| **Versão em produção** | última versão registrada (alias `@production`); a partir da Etapa 3 o artefato é uma `sklearn.Pipeline` empacotada |
| **Algoritmo servido** | Logistic Regression (sklearn) servida como `sklearn.Pipeline(FeatureEngineer → StandardScaler → LogReg)` empacotada num único artefato MLflow + threshold de negócio |
| **Modelo alternativo (A/B-testável)** | MLP (PyTorch). Registrado como `Churn_MLP_Final_Production` v12, selecionável via `MODEL_FLAVOR=pytorch` no `.env` (ver §7.1) |
| **Data de treino** | Q1/2026 |
| **Owner / responsável** | Equipe FIAP MLET (Fase 1) |
| **Tracking** | DagsHub MLflow em `https://dagshub.com/JosueJNLui/fiap-mlet-challenge-fase-1.mlflow` |

### Hiperparâmetros da LogReg servida

- **Solver:** `lbfgs`.
- **Regularização:** L2 (default), `C=1.0`.
- **Tratamento de desbalanceamento:** `class_weight='balanced'`.
- **`max_iter`:** 1000.
- **`random_state`:** 42.

A escolha da LogReg como modelo servido decorre da equivalência estatística com os MLPs do top-3 no K-Fold pareado (Friedman seguido de Nemenyi, p ∈ [0.982, 1.000] vs LogReg, todos não significativos), somada à liderança em lucro de hold-out (R$ 81.200 vs R$ 76.300 do MLP) e à vantagem operacional em **interpretabilidade, parsimônia e auditabilidade** (ver §5 e a célula de promoção em [`notebooks/models-comparison.ipynb`](../notebooks/models-comparison.ipynb)).

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

- **Fonte:** [Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
- **Volume:** 7.043 clientes × 21 colunas.
- **Target:** `Churn` (Yes/No), desbalanceado: 73% No / 27% Yes (5.174 / 1.869 no bruto).

### Qualidade

- 11 valores inválidos detectados em `TotalCharges` (strings em branco para `tenure=0`).
- Tratamento no EDA (`notebooks/eda.ipynb`): imputação `MonthlyCharges × tenure` (mantém 7.043 linhas).
- Tratamento na modelagem (`notebooks/modeling.ipynb`): `FeatureEngineer.transform` faz a mesma imputação `MonthlyCharges × tenure` e em seguida aplica `log1p(TotalCharges)`, então o split estratificado opera sobre as 7.043 linhas (sem `dropna`). Há também uma célula descritiva paralela em `modeling.ipynb` que mostra o efeito de `dropna(TotalCharges)` (resultando 7.032 linhas com distribuição {0: 5.163, 1: 1.869}); ela é puramente exploratória e não alimenta o pipeline de produção.
- Sem duplicatas.
- Sem missing values críticos após o tratamento de `TotalCharges`.

### Splits e validação

- Hold-out estratificado: 80% treino+validação / 20% teste sobre 7.043 linhas, totalizando 5.634 desenvolvimento e 1.409 teste, `random_state=42`.
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

A engenharia de features vive em `src/application/transformers.py::FeatureEngineer` (`BaseEstimator` + `TransformerMixin`) e é orquestrada pelo `sklearn.Pipeline` montado em `src/application/pipeline.py::build_logreg_pipeline`. O fluxo, aplicado por `FeatureEngineer.transform`:

1. Encoding binário (Yes/No → 1/0) em 11 colunas.
2. `InternetService`: DSL/Fiber → 1, No → 0.
3. `TotalCharges`: imputa NaN como `MonthlyCharges × tenure` (clientes com `tenure=0`) e aplica `log1p`.
4. **Feature engineering:**
   - `tenure_bucket` (binning: 0-12, 13-24, 25-48, 49+).
   - `avg_charges_per_month = TotalCharges / tenure`.
   - `charge_vs_expected = MonthlyCharges - avg_charges_per_month`.
   - `num_services` (soma dos 9 serviços contratados).
5. One-hot encoding manual: `gender`, `Contract`, `PaymentMethod`, `tenure_bucket`.
6. `reindex` para `EXPECTED_FEATURE_ORDER` (28 features fixas). É a **fonte única da verdade** entre treino e produção.
7. `StandardScaler` é o segundo step da Pipeline; ajustado no fit do bundle e serializado dentro do mesmo artefato sklearn (não há mais download separado de `scaler.joblib` no flavor sklearn).

### Validação de schema (pandera)

`src/application/data_schemas.py` define dois `DataFrameSchema`:

- `RAW_TELCO_SCHEMA`: 21 colunas do CSV bruto, com tipos, vocabulários e ranges. Usado em `notebooks/eda.ipynb` como passo formal de *data readiness*.
- `PROCESSED_FEATURES_SCHEMA`: 28 colunas float em ordem fixa, validando a saída de `preprocess_one`/`FeatureEngineer.transform`. Usado em testes (`tests/application/test_data_schemas.py`).

Os schemas não são chamados no hot-path da API (overhead desnecessário em payloads de 1 linha), mas firmam o contrato testado entre treino e inferência.

---

## 5. Métricas de Performance

### Métricas técnicas (hold-out, 20% teste)

| Modelo | ROC-AUC | PR-AUC | F1 (churn) | Recall | Precision | Accuracy |
|---|---|---|---|---|---|---|
| DummyClassifier (maj.) | 0.500 | 0.633 | 0.000 | 0.000 | 0.000 | 0.735 |
| **Logistic Regression** | **0.849** | **0.672** | **0.560** | **0.960** | **0.395** | 0.600 |
| Random Forest (100, depth=10) | 0.840 | 0.645 | 0.551 | 0.955 | 0.387 | 0.587 |
| XGBoost | 0.838 | 0.648 | 0.552 | 0.952 | 0.388 | 0.589 |
| MLP (campeão grid search) | 0.849 | – | 0.550 | 0.949 | 0.387 | 0.588 |

> **Nota sobre Accuracy:** os valores baixos (0.59 a 0.60) refletem o threshold otimizado por **lucro líquido** (~0.21), não por F1. Como cada FN custa 5× mais que cada FP, o ponto de operação favorece recall e sacrifica accuracy. O Dummy mantém accuracy alta (0.735) porque sempre prediz a classe majoritária. A célula consolidadora de `modeling.ipynb` que alimenta esta tabela agrega no DataFrame final Dummy, LogReg, RandomForest e XGBoost com PR-AUC; o MLP entra na mesma tabela pelo run `MLP_GridSearch_KFold` mas sem coluna PR-AUC consolidada, por isso aparece como `–` aqui. Os PR-AUC do MLP por fold ficam disponíveis nos runs MLflow individuais (experimento `Churn-Predict-Telco-Etapa2-Modelagem`).

### Métrica de negócio: Lucro Líquido (BRL)

Definida em conjunto com a área:

```
Lucro = TP × LTV  −  FP × Custo_retencao  −  FN × LTV
        com LTV = R$ 500 e Custo_retencao = R$ 100
```

> **Auditabilidade:** o cálculo é centralizado em [`src/application/business_metrics.py`](../src/application/business_metrics.py) e consumido pelos três notebooks (`eda`, `modeling`, `models-comparison`). Qualquer ajuste de LTV, custo de retenção ou da fórmula passa por esse módulo + teste em [`tests/application/test_business_metrics.py`](../tests/application/test_business_metrics.py).

| Modelo | Lucro K-Fold | Lucro hold-out | FP × R$100 | FN × R$500 |
|---|---|---|---|---|
| DummyClassifier | - | **−R$ 187.000** | - | R$ 187.000 |
| **Logistic Regression** | R$ 33.120 | **R$ 81.200** | R$ 54.900 | R$ 7.500 |
| Random Forest | R$ 32.340 | R$ 77.800 | R$ 56.500 | R$ 8.500 |
| XGBoost | R$ 32.360 | R$ 77.300 | R$ 56.100 | R$ 9.000 |
| MLP (top-1) | R$ 32.980 | R$ 76.300 | R$ 56.200 | R$ 9.500 |

### Threshold de decisão (otimizado para negócio)

- **Threshold servido:** `0.2080` (LogReg, otimizado pela curva de lucro na célula `Pipeline de produção` de `notebooks/modeling.ipynb`; persistido como tag `threshold_servido` na versão `@production` do modelo MLflow).
- **Threshold do MLP alternativo:** `0.20303` (mediana dos thresholds ótimos por fold no K-Fold 10-fold de `notebooks/modeling.ipynb`; valor numérico exato `0.203030303030303`).
- Cada threshold foi selecionado varrendo a curva PR do **seu próprio modelo** e maximizando o **lucro líquido** em validação (não o F1). Por isso são distintos.
- Reflete a assimetria de custo: cada FN custa 5× mais que cada FP (LTV R$ 500 vs custo retenção R$ 100), então ambos são deliberadamente baixos para favorecer recall.
- **Por que não usar o ótimo do hold-out do MLP?** O notebook reporta que o threshold ótimo do hold-out do MLP seria `0.1800`, com lucro `R$ 79.400`. Mantemos `0.20303` (mediana K-Fold) por ser estimativa de validação cruzada robusta, com lucro hold-out `R$ 76.300`. Tunar threshold em um split único favorece overfitting, então a mediana K-Fold ganha.
- O baseline da Etapa 1 (`notebooks/eda.ipynb`) usou um threshold inicial `0.2278` (CV 5-fold) que rendeu lucro hold-out de R$ 80.200. A otimização final por curva de lucro da Etapa 3 trouxe o valor para `0.2080` (R$ 81.200), é este último que vai para produção.

### Comparação estatística (Friedman + Nemenyi, 10-fold pareado)

- **Friedman global** (LogReg, RandomForest, XGBoost e top-3 MLPs por lucro K-Fold): estatística `30.8280`, p-valor `≈ 2.73e-05`. Diferenças entre os modelos são significativas.
- **Post-hoc Nemenyi** (LogReg vs cada MLP do top-3 K-Fold): p ∈ [0.982, 1.000], todos não significativos. Por modelo: `MLP_32_lr0.001_drp0.0_b128` p=`0.999975`, `MLP_8_lr0.001_drp0.3_b128` p=`0.998617`, `MLP_8_lr0.01_drp0.3_b128` p=`0.982115`. Conclusão: LogReg é estatisticamente equivalente aos MLPs do topo.
- **Sobre o MLP servido (`hidden=8, lr=0.01, dropout=0.15, batch=128`):** ficou em 4º lugar por lucro K-Fold (R$ 32.980, contra R$ 33.130 a R$ 33.340 do top-3) e por isso não entra na matriz Nemenyi. A proximidade de lucro com os 3 MLPs estatisticamente equivalentes preserva a equivalência prática para fins de A/B test.
- **Decisão:** servir Logistic Regression (parsimônia, interpretabilidade), com MLP versionado como alternativa A/B-testável (ver §7.1).

> **Fonte dos números desta seção:** os resultados de Dummy/LogReg vêm das células finais de [`notebooks/eda.ipynb`](../notebooks/eda.ipynb); os de MLP/RandomForest/XGBoost de [`notebooks/modeling.ipynb`](../notebooks/modeling.ipynb) e [`notebooks/models-comparison.ipynb`](../notebooks/models-comparison.ipynb). Em caso de divergência, **os notebooks são autoritativos**. Este Model Card replica os valores publicados.

---

## 6. Análise de Vieses e Subgrupos

Insights de `notebooks/eda.ipynb` (análise bivariada com Mann-Whitney p<0.05 nas variáveis numéricas):

| Subgrupo | Taxa de churn | Risco relativo | Comentário |
|---|---|---|---|
| `Contract = Month-to-month` | **42,7%** | 🔴 muito alto | Maior driver isolado de churn |
| `Contract = Two year` | 2,8% | 🟢 muito baixo | Cliente "âncora" |
| `InternetService = Fiber optic` | 41,9% | 🔴 alto | Possível insatisfação com produto |
| `InternetService = No` | 7,4% | 🟢 baixo | - |
| `PaymentMethod = Electronic check` | 45,3% | 🔴 muito alto | Forte sinal de menor engajamento |
| `PaperlessBilling = Yes` | 33,6% | 🟡 elevado | - |
| `tenure < 12 meses` | 47,4% | 🔴 muito alto | Janela crítica de onboarding |
| `tenure ≥ 49 meses` | 6,6% | 🟢 muito baixo | - |

### Vieses esperados / a monitorar

- **Concentração em Month-to-month + Electronic check + tenure baixo:** o modelo aprende fortemente esses padrões. Clientes que migrem para essas categorias serão imediatamente sinalizados. Efeito desejado, mas pode causar **falsos positivos** em clientes recém-adquiridos via campanhas que naturalmente começam em mensal.
- **Gênero:** distribuição balanceada (Female ≈ Male) e Mann-Whitney não-significativo na análise → **não esperamos viés de gênero**, mas **não há análise formal de fairness por subgrupo** (gap conhecido, ver §8).
- **Idade (`SeniorCitizen`):** subgrupo é minoria (~16% do dataset). O modelo pode estar sub-otimizado para sêniores. Recomenda-se monitorar Precision/Recall específico desse grupo.
- **Multicolinearidade:** `TotalCharges × tenure` correlação 0.826. Tratada com `log1p(TotalCharges)` + features derivadas, mas Logistic Regression mantém alguma redundância.

---

## 7. Cenários de Falha Esperados

| Cenário | Sintoma | Mitigação |
|---|---|---|
| **Drift de produto** (lançamento de novo plano) | Distribuição de `Contract`, `MonthlyCharges` fora do treino | Retreino; até lá, monitorar drift via KS test (ver `MONITORING.md`) |
| **Drift sazonal** (campanhas de Black Friday, fim de ano) | Pico de assinaturas Month-to-month e churn aumenta | Avaliar ROC-AUC mensal em amostra rotulada; reconfigurar threshold se necessário |
| **Mudança de público-alvo** (ex.: passar a vender B2B) | Modelo prediz churn alto consistentemente | Não usar; o modelo não foi treinado para B2B |
| **Falha no carregamento do modelo** (sklearn) | Pipeline não desserializa (versão de sklearn incompatível, classe `FeatureEngineer` ausente, etc.) | Lifespan fail-fast no startup; CI testa `load_predictor` mockado (`tests/infrastructure/`) e real (`tests/integration/`) |
| **Falha no carregamento do scaler** (apenas pytorch) | MLP usa scaler salvo como `joblib` em `model_components/`. Se download do artefato falha, predições degeneram | Lifespan fail-fast; o flavor sklearn é imune (Pipeline empacotada) |
| **Indisponibilidade do MLflow / DagsHub** | API falha ao subir | `MODEL_VERSION` pinado em env permite rollback determinístico; cache local opcional |
| **Necessidade de A/B test ou rollback do flavor** | Performance da LogReg degrada em segmento específico, ou queremos comparar com MLP | Trocar `MODEL_FLAVOR` / `MODEL_NAME` / `MODEL_VERSION` / `PREDICTION_THRESHOLD` no `.env` e reiniciar (sem deploy de código, ver §7.1) |
| **Payload mal-formado** | Erro 422 (Pydantic) | Validação automática antes de qualquer inferência |

### 7.1 Considerações operacionais: modelo servido vs alternativa

A API mantém **dois caminhos de inferência** (sklearn LogReg e PyTorch MLP) selecionáveis via `settings.model_flavor`. A LogReg é o default por parsimônia somada à equivalência estatística com o top-3 dos MLPs no Nemenyi (p ∈ [0.982, 1.000], ver §5). O MLP fica versionado como alternativa A/B-testável, útil em três cenários: (i) comparação prospectiva de performance em produção, (ii) rollback rápido se a LogReg degradar em algum segmento, (iii) análises pontuais que se beneficiem da capacidade não-linear.

**Receita de troca** (sem deploy de código): editar o `.env` para o bloco "Fallback A/B-testável: MLP (PyTorch)" descrito em [`.env.example`](../.env.example) e reiniciar a API. Os invariantes que **não** mudam: lógica de feature engineering (`FeatureEngineer.transform` aplicada em ambos os caminhos), ordem das 28 features (`EXPECTED_FEATURE_ORDER`) e threshold otimizado por curva de lucro. O que muda por flavor: no sklearn, o `StandardScaler` vive dentro da `sklearn.Pipeline` (artefato único); no pytorch, ele é baixado separadamente de `model_components/scaler.joblib`. O carregamento é dispatchado em [`src/infrastructure/mlflow_loader.py`](../src/infrastructure/mlflow_loader.py).

---

## 8. Limitações Conhecidas

- **Dataset estático (Kaggle).** Sem feed contínuo, não captura tendências reais de mercado.
- **Sem features temporais/comportamentais** (uso mensal, tickets de suporte, NPS).
- **Sem calibração formal de probabilidades.** O `churn_probability` retornado é o score do sigmoid: útil para ranking, mas não como probabilidade verdadeira sem reliability diagram.
- **Sem análise formal de fairness por subgrupo** (gênero, faixa etária, etnia, esta última ausente do dataset).
- **Threshold único global.** Não há thresholds segmentados por subgrupo, o que pode gerar desbalanço operacional (ex.: 90% do "high-risk pool" sendo Month-to-month).
- **Sem mecanismo de feedback loop** para o resultado real da retenção (cliente realmente saiu?), o que limita o monitoramento de negócio (ver `MONITORING.md`).
- **Assimetria de empacotamento entre flavors.** A LogReg de produção é servida como `sklearn.Pipeline` empacotada (1 artefato). O MLP A/B-testável continua como modelo PyTorch + `scaler.joblib` separado (2 artefatos), porque `torch.nn.Module` não cabe nativamente em `sklearn.Pipeline`. Aceito por escopo. Adaptar via `skorch` ou wrapper custom é uma evolução futura.

---

## 9. Considerações Éticas

- O dataset não contém atributos sensíveis explícitos (raça/etnia, religião, nacionalidade), porém `gender` está presente.
- O modelo é uma **ferramenta de apoio**, não decisor automático. Toda ação ofensiva (ex.: contato comercial, oferta) deve passar por revisão humana e regras de negócio.
- Logging estruturado preserva `request_id` mas não persiste payloads, alinhado com princípio de minimização de dados (LGPD/GDPR).

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

- [`notebooks/eda.ipynb`](../notebooks/eda.ipynb): **Etapa 1**, EDA completa, baselines (DummyClassifier, Logistic Regression), métrica de negócio. Experimento: `Churn-Predict-Telco-Etapa1-EDA`.
- [`notebooks/modeling.ipynb`](../notebooks/modeling.ipynb): **Etapa 2**, MLP em PyTorch (grid search 54 combinações, early stopping, K-Fold), ensembles (RandomForest, XGBoost), threshold otimizado, trade-off FP×FN. Experimento: `Churn-Predict-Telco-Etapa2-Modelagem`.
- [`notebooks/models-comparison.ipynb`](../notebooks/models-comparison.ipynb): **Etapas 1+2**, leitura cruzada dos dois experimentos, ranking por lucro, análise de estabilidade validação vs. teste.

---

## 11. Documentos Relacionados

- [`ARCHITECTURE_DEPLOY.md`](ARCHITECTURE_DEPLOY.md): arquitetura de deploy (real-time vs batch, SLA, scaling).
- [`MONITORING.md`](MONITORING.md): plano de monitoramento (técnico, modelo, negócio, alertas, playbook).
- [`../README.md`](../README.md): instruções de setup e uso da API.
