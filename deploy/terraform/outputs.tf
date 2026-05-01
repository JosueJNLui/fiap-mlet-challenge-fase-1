output "endpoint_url" {
  description = "Public HTTP endpoint for the application."
  value       = "http://${aws_lb.main.dns_name}"
}

output "alb_dns_name" {
  description = "DNS name of the public ALB."
  value       = aws_lb.main.dns_name
}

output "ecs_cluster_name" {
  description = "ECS cluster name."
  value       = aws_ecs_cluster.main.name
}

output "ecs_service_name" {
  description = "ECS service name."
  value       = aws_ecs_service.app.name
}
