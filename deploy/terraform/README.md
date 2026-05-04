# Terraform ECS deploy

Deploy de baixo custo na AWS usando ECS Fargate Spot, 2 tasks pequenas e um
Application Load Balancer público.

## Recursos criados

- VPC simples com 2 subnets públicas e Internet Gateway.
- ECS Cluster com `FARGATE_SPOT`.
- ECS Service com 2 tasks `256 CPU / 512 MiB` (a API requer pelo menos 380 MiB;
  512 MiB é o menor tamanho válido do Fargate para 256 CPU).
- Application Load Balancer HTTP na porta 80.
- CloudWatch Log Group com retenção de 7 dias.

Não há NAT Gateway para reduzir custo. As tasks recebem IP público para baixar a
imagem do GHCR e acessar o MLflow/DagsHub.

## Variáveis via ambiente

Configure as credenciais AWS usando o fluxo padrão do provider:

```bash
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_DEFAULT_REGION="us-east-1"
```

O deploy injeta os mesmos defaults de runtime do `.env.example`, incluindo
`MLFLOW_TRACKING_URI`, `MODEL_FLAVOR`, `MODEL_NAME`, `MODEL_VERSION`,
`SCALER_ARTIFACT_PATH`, `PREDICTION_THRESHOLD`, `LOAD_MODEL_ON_STARTUP`,
`DOCS_URL`, `REDOC_URL` e `OPENAPI_URL`. Para autenticação no registry, informe
`mlflow_tracking_password_secret_arn` com o ARN de um secret/parameter contendo
o token, evitando colocar o valor sensível no state do Terraform.

## Como aplicar

Na primeira execução, crie o bucket remoto de state e gere o arquivo local
`backend.hcl`:

```bash
cd deploy/terraform
python3 scripts/bootstrap_backend.py
```

O script usa as credenciais AWS do ambiente, cria um bucket S3 com versionamento,
criptografia e bloqueio de acesso público, e escreve `backend.hcl`. Para
customizar:

```bash
export TF_STATE_BUCKET="meu-bucket-de-state"
export TF_STATE_REGION="us-east-1"
export TF_STATE_KEY="fiap-mlet/terraform.tfstate"
python3 scripts/bootstrap_backend.py
```

Depois inicialize o backend S3:

```bash
cd deploy/terraform
terraform init -backend-config=backend.hcl
terraform plan
terraform apply
```

Ao final, use o output `endpoint_url` para chamar a API:

```bash
curl "$(terraform output -raw endpoint_url)/health"
```

Observação: se o registry deixar de ser público, adicione as variáveis de auth
como secrets do ECS sem colocar o valor sensível no state do Terraform.
