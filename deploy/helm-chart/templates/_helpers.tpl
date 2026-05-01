{{/*
Expand the name of the chart.
*/}}
{{- define "fiap-mlet.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "fiap-mlet.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "fiap-mlet.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Common labels.
*/}}
{{- define "fiap-mlet.labels" -}}
helm.sh/chart: {{ include "fiap-mlet.chart" . }}
{{ include "fiap-mlet.selectorLabels" . }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{/*
Selector labels.
*/}}
{{- define "fiap-mlet.selectorLabels" -}}
app.kubernetes.io/name: {{ include "fiap-mlet.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{/*
Create the name of the service account to use.
*/}}
{{- define "fiap-mlet.serviceAccountName" -}}
{{- if .Values.serviceAccount.create -}}
{{- default (include "fiap-mlet.fullname" .) .Values.serviceAccount.name -}}
{{- else -}}
{{- default "default" .Values.serviceAccount.name -}}
{{- end -}}
{{- end -}}

{{/*
Secret name for sensitive environment variables.
*/}}
{{- define "fiap-mlet.secretName" -}}
{{- default (printf "%s-env" (include "fiap-mlet.fullname" .)) .Values.secretEnv.name -}}
{{- end -}}
