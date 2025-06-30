#!/bin/bash

# XGBoost Ops Setup Script
# This script installs all prerequisites and prepares the environment

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
    if command -v $1 &> /dev/null; then
        print_success "$1 is already installed"
        return 0
    else
        print_warning "$1 is not installed"
        return 1
    fi
}

# Function to install Homebrew packages
install_brew_package() {
    local package=$1
    if ! brew list $package &> /dev/null; then
        print_status "Installing $package..."
        brew install $package
        print_success "$package installed successfully"
    else
        print_success "$package is already installed"
    fi
}

# Function to install Python packages
install_python_packages() {
    print_status "Installing Python packages..."
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install packages from requirements.txt
    pip install -r requirements.txt
    
    print_success "Python packages installed successfully"
}

# Function to configure AWS CLI
configure_aws() {
    print_status "Configuring AWS CLI..."
    
    if ! aws sts get-caller-identity &> /dev/null; then
        print_warning "AWS CLI not configured. Please run 'aws configure' manually."
        print_status "You'll need to provide:"
        print_status "  - AWS Access Key ID"
        print_status "  - AWS Secret Access Key"
        print_status "  - Default region (e.g., us-west-2)"
        print_status "  - Default output format (json)"
    else
        print_success "AWS CLI is already configured"
    fi
}

# Function to create environment file
create_env_file() {
    print_status "Creating .env file..."
    
    cat > .env << EOF
# AWS Configuration
AWS_DEFAULT_REGION=us-west-2
S3_BUCKET_NAME=xgboost-ops-models

# MLflow Configuration
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_DEFAULT_ARTIFACT_ROOT=s3://xgboost-ops-models/mlflow-artifacts

# Model Configuration
MODEL_NAME=xgboost-model
MODEL_VERSION=latest

# Kubernetes Configuration
KUBERNETES_NAMESPACE=xgboost-serving
EOF

    print_success ".env file created"
}

# Main setup function
main() {
    print_status "🚀 Starting XGBoost Ops setup..."
    
    # Check if we're on macOS
    if [[ "$OSTYPE" != "darwin"* ]]; then
        print_error "This setup script is designed for macOS. Please adapt for your OS."
        exit 1
    fi
    
    # Check if Homebrew is installed
    if ! command -v brew &> /dev/null; then
        print_error "Homebrew is not installed. Please install Homebrew first:"
        print_status "Visit: https://brew.sh"
        exit 1
    fi
    
    print_success "Homebrew is installed"
    
    # Install system dependencies
    print_status "Installing system dependencies..."
    
    install_brew_package "python@3.11"
    install_brew_package "terraform"
    install_brew_package "kubectl"
    install_brew_package "helm"
    install_brew_package "awscli"
    install_brew_package "docker"
    
    # Install Python packages
    install_python_packages
    
    # Configure AWS CLI
    configure_aws
    
    # Create environment file
    create_env_file
    
    # Create necessary directories
    print_status "Creating project directories..."
    mkdir -p models data logs
    
    # Make scripts executable
    print_status "Making scripts executable..."
    chmod +x scripts/*.sh scripts/*.py
    
    # Final setup summary
    echo ""
    print_success "🎉 XGBoost Ops setup completed successfully!"
    echo ""
    echo "📋 Next steps:"
    echo "   1. Configure AWS credentials: aws configure"
    echo "   2. Start Docker Desktop"
    echo "   3. Run the deployment: ./scripts/deploy.sh"
    echo ""
    echo "🔧 Available commands:"
    echo "   ./scripts/deploy.sh          - Deploy complete infrastructure"
    echo "   python scripts/train_model.py - Train XGBoost model"
    echo "   python notebooks/xgboost_training.py - Interactive training"
    echo "   kubectl get pods -A          - Check Kubernetes pods"
    echo ""
    echo "📚 Documentation:"
    echo "   README.md                    - Complete deployment guide"
    echo "   terraform/                   - Infrastructure as Code"
    echo "   k8s/                        - Kubernetes manifests"
    echo "   scripts/                     - Deployment scripts"
    echo ""
    echo "⚠️  Important notes:"
    echo "   - Make sure Docker Desktop is running"
    echo "   - Ensure AWS credentials are configured"
    echo "   - The deployment will create AWS resources (costs may apply)"
    echo "   - Monitor your AWS billing dashboard"
}

# Run main function
main "$@" 