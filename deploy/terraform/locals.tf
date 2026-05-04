locals {
  tags = merge(
    {
      Project     = var.project_name
      ManagedBy   = "terraform"
      Environment = "single"
      application = "fiap-mlet-challenge-fase-1"
    },
    var.tags
  )

  availability_zones = slice(data.aws_availability_zones.available.names, 0, 2)

  container_environment = [
    {
      name  = "MLFLOW_TRACKING_URI"
      value = var.mlflow_tracking_uri
    },
    {
      name  = "MLFLOW_TRACKING_USERNAME"
      value = var.mlflow_tracking_username
    },
    {
      name  = "MODEL_FLAVOR"
      value = var.model_flavor
    },
    {
      name  = "MODEL_NAME"
      value = var.model_name
    },
    {
      name  = "MODEL_VERSION"
      value = var.model_version
    },
    {
      name  = "SCALER_ARTIFACT_PATH"
      value = var.scaler_artifact_path
    },
    {
      name  = "PREDICTION_THRESHOLD"
      value = var.prediction_threshold
    },
    {
      name  = "LOAD_MODEL_ON_STARTUP"
      value = tostring(var.load_model_on_startup)
    },
    {
      name  = "DOCS_URL"
      value = var.docs_url
    },
    {
      name  = "REDOC_URL"
      value = var.redoc_url
    },
    {
      name  = "OPENAPI_URL"
      value = var.openapi_url
    }
  ]

  container_secrets = var.mlflow_tracking_password_secret_arn == "" ? [] : [
    {
      name      = "MLFLOW_TRACKING_PASSWORD"
      valueFrom = var.mlflow_tracking_password_secret_arn
    }
  ]
}

data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_region" "current" {}

data "aws_caller_identity" "current" {}
