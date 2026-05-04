
# 📜 Diretrizes do Projeto: Python DDD & Clean Code

Este arquivo serve como a única fonte de verdade para padrões de código, arquitetura e fluxo de trabalho. **Siga estas regras rigorosamente.**

## 🚀 1. Stack Tecnológica & Gestão
- **Linguagem:** Python 3.13.+ (Uso intensivo de novos recursos de performance).
- **Web Framework:** FastAPI 0.136+ (Aproveitar tipagem avançada e Pydantic v2).
- **Gestão de Dependências:** **Poetry** (Gerenciar `pyproject.toml` e `poetry.lock`).
- **Performance de Ambiente:** **uv** (Utilizado para `sync`, `install` e download de pacotes).
- **Type Checker:** `ty` (Apenas ele deve ser usado para verificação estática).
- **Linter/Formatter:** `ruff` (Executar `ruff check --fix` e `ruff format`).

## 🏛️ 2. Arquitetura: DDD & Clean Code
O projeto é orientado a **Domain-Driven Design**. Mantenha a separação de interesses:

1.  **Domain:** Lógica pura, entidades e Value Objects. **Proibido** importar bibliotecas externas aqui (exceto tipagem).
2.  **Application:** Casos de uso e serviços que orquestram o domínio.
3.  **Infrastructure:** Implementação de repositórios, persistência e integrações externas.
4.  **Interface/API:** Endpoints FastAPI, Middlewares e Schemas (Pydantic).

**Clean Code:**
- Nomes altamente descritivos.
- Injeção de dependência via construtores.
- Funções atômicas (Single Responsibility Principle).

## 🛠️ 3. Regras de Implementação & Qualidade

### Tipagem e Formatação
- **`ty` Check:** O código deve estar 100% tipado e passar no `ty`.
- **`ruff` Style:** Formatação automática em cada salvamento. Siga rigorosamente a PEP 8 via Ruff.

### Testes (Pytest)
Sempre que criar uma feature, os seguintes testes são obrigatórios:
- **Unitários:** Cobertura total da camada de Domínio e Casos de Uso.
- **Schema (Pandera):** Validação de integridade de dados complexos ou DataFrames.
- **Smoke Tests:** Validar se os endpoints críticos do FastAPI estão acessíveis.

### Observabilidade & Middleware
- **Middleware de Latência:** Implementar log de `X-Process-Time` em todos os requests.
- **Logging Estruturado:** Logs em formato JSON contendo `correlation_id`, `level`, `timestamp` e `message`.

## 📦 4. Fluxo de Gerenciamento (Poetry + uv)
Ao adicionar ou atualizar o projeto:
1.  Use virtual env. `uv venv`, para ativcar `.venv/bin/activate`.
2.  Use `poetry add <package>` para atualizar o `pyproject.toml`.
3.  Use o **`uv`** para sincronizar o ambiente virtual instantaneamente: `uv pip install -r <(poetry export -f requirements.txt)` ou, se preferir o workflow direto, `uv sync` (se configurado com o poetry).
4.  **Nota:** Priorize o `uv` para qualquer operação de download e sync por ser significativamente mais rápido.