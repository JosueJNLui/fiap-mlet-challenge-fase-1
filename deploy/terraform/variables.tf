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
  default     = "ghcr.io/josuejnlui/fiap-mlet-challenge-fase-1:fix-ouput-logs"
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
  description = "Fargate task memory in MiB."
  type        = number
  default     = 512
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

variable "model_name" {
  description = "Registered model name loaded by the API."
  type        = string
  default     = "Churn_MLP_Final_Production"
}

variable "model_version" {
  description = "Registered model version loaded by the API."
  type        = string
  default     = "8"
}

variable "prediction_threshold" {
  description = "Prediction threshold used by the API."
  type        = string
  default     = "0.20303030303030303"
}

variable "load_model_on_startup" {
  description = "Whether the API loads the ML model during startup."
  type        = bool
  default     = true
}

variable "tags" {
  description = "Extra tags added to all supported resources."
  type        = map(string)
  default     = {}
}
