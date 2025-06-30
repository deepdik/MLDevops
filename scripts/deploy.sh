#!/bin/bash

# XGBoost Model Deployment Script
# This script deploys the complete XGBoost model infrastructure on AWS

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
check_command() {
    if ! command -v $1 &> /dev/null; then
        print_error "$1 is not installed. Please install $1 first."
        exit 1
    fi
}

# Function to check AWS credentials
check_aws_credentials() {
    if ! aws sts get-caller-identity &> /dev/null; then
        print_error "AWS credentials not configured. Please run 'aws configure' first."
        exit 1
    fi
}

# Function to create AWS credentials secret
create_aws_secret() {
    print_status "Creating AWS credentials secret..."
    
    # Get AWS credentials
    AWS_ACCESS_KEY_ID=$(aws configure get aws_access_key_id)
    AWS_SECRET_ACCESS_KEY=$(aws configure get aws_secret_access_key)
    
    # Base64 encode credentials
    AWS_ACCESS_KEY_ID_B64=$(echo -n "$AWS_ACCESS_KEY_ID" | base64)
    AWS_SECRET_ACCESS_KEY_B64=$(echo -n "$AWS_SECRET_ACCESS_KEY" | base64)
    
    # Create secret
    cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Secret
metadata:
  name: aws-credentials
  namespace: xgboost-serving
type: Opaque
data:
  aws-access-key-id: $AWS_ACCESS_KEY_ID_B64
  aws-secret-access-key: $AWS_SECRET_ACCESS_KEY_B64
---
apiVersion: v1
kind: Secret
metadata:
  name: seldon-init-container-secret
  namespace: xgboost-serving
type: Opaque
data:
  AWS_ACCESS_KEY_ID: $AWS_ACCESS_KEY_ID_B64
  AWS_SECRET_ACCESS_KEY: $AWS_SECRET_ACCESS_KEY_B64
  AWS_DEFAULT_REGION: dXMtZWFzdC0y
EOF
    
    print_success "AWS credentials secret created"
}

# Main deployment function
main() {
    print_status "Starting XGBoost model deployment on AWS..."
    
    # Check prerequisites
    print_status "Checking prerequisites..."
    check_command "aws"
    check_command "kubectl"
    check_command "helm"
    check_command "terraform"
    check_aws_credentials
    
    print_success "All prerequisites are satisfied"
    
    # Step 1: Deploy infrastructure with Terraform
    print_status "Step 1: Deploying infrastructure with Terraform..."
    cd terraform
    
    print_status "Initializing Terraform..."
    terraform init
    
    print_status "Planning Terraform deployment..."
    terraform plan -out=tfplan
    
    print_status "Applying Terraform configuration..."
    terraform apply tfplan
    
    # Get outputs
    CLUSTER_NAME=$(terraform output -raw cluster_name)
    CLUSTER_ENDPOINT=$(terraform output -raw cluster_endpoint)
    MODEL_BUCKET=$(terraform output -raw model_bucket_name)
    
    print_success "Infrastructure deployed successfully"
    cd ..
    
    # Step 2: Configure kubectl for EKS cluster
    print_status "Step 2: Configuring kubectl for EKS cluster..."
    aws eks update-kubeconfig --region us-east-2 --name $CLUSTER_NAME
    
    print_success "kubectl configured for EKS cluster"
    
    # Step 3: Install Kubeflow
    print_status "Step 3: Installing Kubeflow..."
    chmod +x scripts/install_kubeflow.sh
    ./scripts/install_kubeflow.sh
    
    print_success "Kubeflow installed successfully"
    
    # Step 4: Create AWS credentials secret
    print_status "Step 4: Creating AWS credentials secret..."
    create_aws_secret
    
    # Step 5: Train XGBoost model
    print_status "Step 5: Training XGBoost model..."
    
    # Set environment variables
    export S3_BUCKET_NAME=$MODEL_BUCKET
    export MLFLOW_TRACKING_URI="http://localhost:5000"
    
    # Train model
    python scripts/train_model.py \
        --n-estimators 100 \
        --max-depth 6 \
        --learning-rate 0.1 \
        --n-samples 10000 \
        --experiment-name xgboost-production
    
    print_success "XGBoost model trained successfully"
    
    # Step 6: Deploy model to Kubernetes
    print_status "Step 6: Deploying model to Kubernetes..."
    
    # Update the model URI in the deployment manifest
    sed -i.bak "s|s3://xgboost-ops-models/models/xgboost_model_latest.json|s3://$MODEL_BUCKET/models/xgboost_model_latest.json|g" k8s/xgboost-deployment.yaml
    
    # Apply the deployment
    kubectl apply -f k8s/xgboost-deployment.yaml
    
    print_success "Model deployed to Kubernetes"
    
    # Step 7: Wait for deployment to be ready
    print_status "Step 7: Waiting for deployment to be ready..."
    kubectl wait --for=condition=ready pod -l seldon-app=xgboost-predictor-default-0 -n xgboost-serving --timeout=300s
    
    print_success "Deployment is ready"
    
    # Step 8: Get service URLs
    print_status "Step 8: Getting service URLs..."
    
    KUBEFLOW_URL=$(kubectl get svc -n istio-system istio-ingressgateway -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
    MODEL_URL=$(kubectl get ingress -n xgboost-serving xgboost-model-ingress -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
    MLFLOW_URL=$(kubectl get ingress -n mlflow mlflow-ingress -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
    
    # Step 9: Test the deployment
    print_status "Step 9: Testing the deployment..."
    python scripts/test_inference.py --model-url "http://$MODEL_URL"
    
    print_success "Deployment test completed"
    
    # Final summary
    echo ""
    print_success "🎉 XGBoost model deployment completed successfully!"
    echo ""
    echo "📋 Service URLs:"
    echo "   Kubeflow Dashboard: http://$KUBEFLOW_URL"
    echo "   MLflow Tracking: http://$MLFLOW_URL"
    echo "   XGBoost Model API: http://$MODEL_URL"
    echo ""
    echo "🔑 Default Kubeflow credentials:"
    echo "   Username: admin@example.com"
    echo "   Password: 12341234"
    echo ""
    echo "📝 Next steps:"
    echo "   1. Access the Kubeflow dashboard to monitor your ML pipeline"
    echo "   2. Use MLflow to track model experiments"
    echo "   3. Send prediction requests to your deployed model"
    echo "   4. Monitor model performance and retrain as needed"
    echo ""
    echo "🔧 Useful commands:"
    echo "   kubectl get pods -n kubeflow"
    echo "   kubectl get pods -n xgboost-serving"
    echo "   kubectl get pods -n mlflow"
    echo "   kubectl logs -n xgboost-serving -l seldon-app=xgboost-predictor-default-0"
}

# Run main function
main "$@" 