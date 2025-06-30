#!/bin/bash

# Kubeflow Installation Script
# This script installs Kubeflow on an existing EKS cluster

set -e

echo "🚀 Starting Kubeflow installation..."

# Check if kubectl is installed
if ! command -v kubectl &> /dev/null; then
    echo "❌ kubectl is not installed. Please install kubectl first."
    exit 1
fi

# Check if helm is installed
if ! command -v helm &> /dev/null; then
    echo "❌ helm is not installed. Please install helm first."
    exit 1
fi

# Check if we can connect to the cluster
if ! kubectl cluster-info &> /dev/null; then
    echo "❌ Cannot connect to Kubernetes cluster. Please configure kubectl first."
    exit 1
fi

echo "✅ Prerequisites check passed"

# Create namespace for Kubeflow
echo "📦 Creating kubeflow namespace..."
kubectl create namespace kubeflow --dry-run=client -o yaml | kubectl apply -f -

# Add Kubeflow Helm repository
echo "📚 Adding Kubeflow Helm repository..."
helm repo add kubeflow https://github.com/kubeflow/manifests/archive/refs/heads/v1.7-branch.tar.gz
helm repo update

# Install Kubeflow
echo "🔧 Installing Kubeflow..."
helm install kubeflow kubeflow/kubeflow \
    --namespace kubeflow \
    --create-namespace \
    --set istio.enabled=true \
    --set dex.enabled=true \
    --set oidc.enabled=false \
    --set istio.gateway.type=LoadBalancer

echo "⏳ Waiting for Kubeflow components to be ready..."
kubectl wait --for=condition=ready pod -l app=istio-ingressgateway -n istio-system --timeout=300s

# Install Seldon Core
echo "🔧 Installing Seldon Core..."
kubectl create namespace seldon-system --dry-run=client -o yaml | kubectl apply -f -

helm repo add seldonio https://storage.googleapis.com/seldon-charts
helm repo update

helm install seldon-core seldonio/seldon-core-operator \
    --namespace seldon-system \
    --set istio.enabled=true \
    --set ambassador.enabled=false

# Install MLflow
echo "🔧 Installing MLflow..."
kubectl create namespace mlflow --dry-run=client -o yaml | kubectl apply -f -

cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlflow
  namespace: mlflow
spec:
  replicas: 1
  selector:
    matchLabels:
      app: mlflow
  template:
    metadata:
      labels:
        app: mlflow
    spec:
      containers:
      - name: mlflow
        image: ghcr.io/mlflow/mlflow:latest
        ports:
        - containerPort: 5000
        env:
        - name: MLFLOW_TRACKING_URI
          value: "sqlite:///mlflow.db"
        - name: MLFLOW_DEFAULT_ARTIFACT_ROOT
          value: "s3://xgboost-ops-models/mlflow-artifacts"
        - name: AWS_ACCESS_KEY_ID
          valueFrom:
            secretKeyRef:
              name: aws-credentials
              key: aws-access-key-id
        - name: AWS_SECRET_ACCESS_KEY
          valueFrom:
            secretKeyRef:
              name: aws-credentials
              key: aws-secret-access-key
        - name: AWS_DEFAULT_REGION
          value: "us-west-2"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
---
apiVersion: v1
kind: Service
metadata:
  name: mlflow-service
  namespace: mlflow
spec:
  selector:
    app: mlflow
  ports:
  - port: 5000
    targetPort: 5000
  type: ClusterIP
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: mlflow-ingress
  namespace: mlflow
  annotations:
    kubernetes.io/ingress.class: "alb"
    alb.ingress.kubernetes.io/scheme: "internet-facing"
    alb.ingress.kubernetes.io/target-type: "ip"
spec:
  rules:
  - http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: mlflow-service
            port:
              number: 5000
EOF

echo "⏳ Waiting for MLflow to be ready..."
kubectl wait --for=condition=ready pod -l app=mlflow -n mlflow --timeout=300s

# Get the external IP/URL
echo "🌐 Getting external endpoints..."

KUBEFLOW_URL=$(kubectl get svc -n istio-system istio-ingressgateway -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
MLFLOW_URL=$(kubectl get ingress -n mlflow mlflow-ingress -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')

echo "✅ Kubeflow installation completed!"
echo ""
echo "📋 Access URLs:"
echo "   Kubeflow Dashboard: http://$KUBEFLOW_URL"
echo "   MLflow Tracking: http://$MLFLOW_URL"
echo ""
echo "🔑 Default credentials:"
echo "   Username: admin@example.com"
echo "   Password: 12341234"
echo ""
echo "📝 Next steps:"
echo "   1. Access the Kubeflow dashboard"
echo "   2. Create a new notebook server"
echo "   3. Train your XGBoost model"
echo "   4. Deploy the model using Seldon Core"
echo ""
echo "🔧 To check the status of all components:"
echo "   kubectl get pods -n kubeflow"
echo "   kubectl get pods -n seldon-system"
echo "   kubectl get pods -n mlflow" 