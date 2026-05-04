variable "aws_region" {
  description = "AWS region where the ECS service will run."
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Name prefix used for AWS resources."
  type        = string
  default     = "fiap-mlet"
}

variable "container_image" {
  description = "Container image used by the ECS task."
  type        = string
  default     = "ghcr.io/josuejnlui/fiap-mlet-challenge-fase-1:latest"
}

variable "desired_count" {
  description = "Number of ECS tasks to keep running."
  type        = number
  default     = 2
}

variable "container_cpu" {
  description = "Fargate task CPU units. 256 equals 0.25 vCPU."
  type        = number
  default     = 256
}

variable "container_memory" {
  description = "Fargate task memory in MiB. The app requires at least 380 MiB; 512 MiB is the minimum valid Fargate size for 256 CPU."
  type        = number
  default     = 512

  validation {
    condition     = var.container_memory >= 380
    error_message = "container_memory must be at least 380 MiB."
  }
}

variable "container_port" {
  description = "Port exposed by the application container."
  type        = number
  default     = 8000
}

variable "health_check_path" {
  description = "HTTP path used by the ALB target group health check."
  type        = string
  default     = "/health"
}

variable "health_check_grace_period_seconds" {
  description = "Seconds ECS waits before evaluating load balancer health checks for a new task."
  type        = number
  default     = 300
}

variable "container_health_check_interval" {
  description = "Seconds between ECS container health check attempts."
  type        = number
  default     = 30
}

variable "container_health_check_timeout" {
  description = "Seconds to wait before an ECS container health check attempt times out."
  type        = number
  default     = 5
}

variable "container_health_check_retries" {
  description = "Consecutive ECS container health check failures required before marking the container unhealthy."
  type        = number
  default     = 3
}

variable "container_health_check_start_period" {
  description = "Startup grace period in seconds before ECS counts container health check failures."
  type        = number
  default     = 120
}

variable "mlflow_tracking_uri" {
  description = "MLflow tracking URI used by the API."
  type        = string
  default     = "https://dagshub.com/JosueJNLui/fiap-mlet-challenge-fase-1.mlflow"
}

variable "mlflow_tracking_username" {
  description = "DagsHub username used by MLflow when authentication is required."
  type        = string
  default     = "JosueJNLui"
}

variable "mlflow_tracking_password_secret_arn" {
  description = "Optional Secrets Manager or SSM Parameter ARN injected as MLFLOW_TRACKING_PASSWORD."
  type        = string
  default     = ""
}

variable "model_flavor" {
  description = "Inference flavor loaded by the API: sklearn or pytorch."
  type        = string
  default     = "sklearn"

  validation {
    condition     = contains(["sklearn", "pytorch"], var.model_flavor)
    error_message = "model_flavor must be either sklearn or pytorch."
  }
}

variable "model_name" {
  description = "Registered model name loaded by the API."
  type        = string
  default     = "Churn_LogReg_Final_Production"
}

variable "model_version" {
  description = "Registered model version loaded by the API."
  type        = string
  default     = "3"
}

variable "scaler_artifact_path" {
  description = "MLflow artifact path for the scaler used by the pytorch flavor."
  type        = string
  default     = "model_components/scaler.joblib"
}

variable "prediction_threshold" {
  description = "Prediction threshold used by the API."
  type        = string
  default     = "0.2080"
}

variable "load_model_on_startup" {
  description = "Whether the API loads the ML model during startup."
  type        = bool
  default     = true
}

variable "docs_url" {
  description = "Swagger UI path. Set to an empty string to disable it."
  type        = string
  default     = "/docs"
}

variable "redoc_url" {
  description = "ReDoc UI path. Set to an empty string to disable it."
  type        = string
  default     = "/redoc"
}

variable "openapi_url" {
  description = "OpenAPI JSON path. Set to an empty string to disable it."
  type        = string
  default     = "/openapi.json"
}

variable "tags" {
  description = "Extra tags added to all supported resources."
  type        = map(string)
  default     = {}
}
