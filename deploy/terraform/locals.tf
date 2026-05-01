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
      name  = "MODEL_NAME"
      value = var.model_name
    },
    {
      name  = "MODEL_VERSION"
      value = var.model_version
    },
    {
      name  = "PREDICTION_THRESHOLD"
      value = var.prediction_threshold
    },
    {
      name  = "LOAD_MODEL_ON_STARTUP"
      value = tostring(var.load_model_on_startup)
    }
  ]
}

data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_region" "current" {}

data "aws_caller_identity" "current" {}
