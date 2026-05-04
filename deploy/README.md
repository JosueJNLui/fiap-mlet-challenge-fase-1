# Deploy

Código de infraestrutura da API de churn. A documentação operacional completa
(visão geral, diagramas, comandos, variáveis de ambiente, checklist de
produção) vive em [`../docs/DEPLOYMENT.md`](../docs/DEPLOYMENT.md).

## Conteúdo desta pasta

| Caminho | Descrição |
|---|---|
| [`helm-chart/`](helm-chart/) | Chart Helm para deploy em Kubernetes (Deployment, Service, Ingress, HPA, Secret). |
| [`helmfile.yaml`](helmfile.yaml) | Release `fiap-mlet` aplicada via `helmfile sync`. |
| [`terraform/`](terraform/) | Stack AWS ECS Fargate Spot + ALB + CloudWatch ([README do módulo](terraform/README.md)). |
