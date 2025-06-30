# MLDevops Deployment Guide

This guide provides step-by-step instructions for deploying the complete MLDevops pipeline on AWS EKS.

## 🎯 Overview

This deployment creates a complete ML infrastructure with:
- **AWS EKS Cluster** with auto-scaling nodes
- **Kubeflow ML Platform** for model training and orchestration
- **Jenkins CI/CD** for automation
- **XGBoost Model Serving** with Flask API
- **S3 Storage** for model artifacts
- **RDS PostgreSQL** for metadata storage

## 📋 Prerequisites

### Required Tools
```bash
# Install required tools
brew install awscli kubectl terraform helm python@3.9

# Verify installations
aws --version
kubectl version --client
terraform --version
helm version
python3 --version
```

### AWS Configuration
```bash
# Configure AWS CLI
aws configure

# Set environment variables
export AWS_REGION=us-east-2
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
```

## 🚀 Step-by-Step Deployment

### Step 1: Clone and Setup Repository
```bash
# Clone the repository
git clone <your-repo-url>
cd xgboostOps

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Infrastructure Deployment
```bash
# Navigate to terraform directory
cd terraform

# Initialize Terraform
terraform init

# Review the plan
terraform plan

# Apply infrastructure
terraform apply -auto-approve

# Get cluster credentials
aws eks update-kubeconfig --region us-east-2 --name ml-devops-cluster
```

### Step 3: Verify EKS Cluster
```bash
# Check cluster status
kubectl cluster-info

# Check nodes
kubectl get nodes

# Check namespaces
kubectl get namespaces
```

### Step 4: Install Kubeflow
```bash
# Run the installation script
./scripts/install_kubeflow.sh

# Monitor installation
kubectl get pods -n kubeflow
```

### Step 5: Deploy Jenkins
```bash
# Deploy Jenkins
kubectl apply -f k8s/jenkins-deployment.yaml

# Wait for Jenkins to be ready
kubectl wait --for=condition=ready pod -l app=jenkins -n jenkins --timeout=300s
```

### Step 6: Train and Deploy Model
```bash
# Train the model
python notebooks/xgboost_training_simple.py

# Deploy model service
kubectl apply -f k8s/xgboost-flask-deployment.yaml

# Wait for deployment
kubectl rollout status deployment/xgboost-model -n kubeflow
```

### Step 7: Setup Port Forwarding
```bash
# Setup port forwarding for all services
kubectl port-forward -n kubeflow svc/centraldashboard 8080:80 &
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8081:80 &
kubectl port-forward -n kubeflow svc/xgboost-model-service 8082:80 &
kubectl port-forward -n jenkins svc/jenkins 8083:80 &
```

## 🔧 Configuration

### Environment Variables
Create a `.env` file:
```bash
AWS_REGION=us-east-2
AWS_ACCOUNT_ID=your-account-id
KUBECONFIG=~/.kube/config
MODEL_NAME=xgboost-model
NAMESPACE=kubeflow
```

### Terraform Variables
Update `terraform/variables.tf` if needed:
```hcl
variable "aws_region" {
  description = "AWS region"
  default     = "us-east-2"
}

variable "cluster_name" {
  description = "EKS cluster name"
  default     = "ml-devops-cluster"
}
```

## 🧪 Testing

### Test Model API
```bash
# Health check
curl http://localhost:8082/health

# Single prediction
curl -X POST http://localhost:8082/predict \
  -H "Content-Type: application/json" \
  -d '{"data": [[1.0, 2.0, 3.0, 4.0, 5.0]]}'

# Batch prediction
curl -X POST http://localhost:8082/predict_batch \
  -H "Content-Type: application/json" \
  -d '{"data": [[1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 3.0, 4.0, 5.0, 6.0]]}'
```

### Test Jenkins Pipeline
1. Access Jenkins at http://localhost:8083/jenkins/
2. Create a new pipeline job
3. Use the Jenkinsfile from the repository
4. Run the pipeline

### Test Kubeflow Automation
```bash
# Deploy automation job
kubectl apply -f k8s/automation-job.yaml

# Check job status
kubectl get jobs -n kubeflow

# View logs
kubectl logs -n kubeflow job/xgboost-automation-job
```

## 📊 Monitoring

### Check Service Status
```bash
# Check all pods
kubectl get pods -A

# Check services
kubectl get svc -A

# Check deployments
kubectl get deployments -A
```

### View Logs
```bash
# Jenkins logs
kubectl logs -n jenkins deployment/jenkins

# Model service logs
kubectl logs -n kubeflow deployment/xgboost-model

# Kubeflow logs
kubectl logs -n kubeflow deployment/centraldashboard
```

## 🛠️ Troubleshooting

### Common Issues

1. **PVC Binding Issues**
   ```bash
   # Check PVC status
   kubectl get pvc -n kubeflow
   
   # Check storage class
   kubectl get storageclass
   
   # Install EBS CSI driver if needed
   kubectl apply -k "github.com/kubernetes-sigs/aws-ebs-csi-driver/deploy/kubernetes/overlays/stable/?ref=release-1.28"
   ```

2. **Resource Constraints**
   ```bash
   # Check node resources
   kubectl top nodes
   
   # Check pod resources
   kubectl top pods -n kubeflow
   
   # Scale up node group if needed
   aws eks update-nodegroup-config \
     --cluster-name ml-devops-cluster \
     --nodegroup-name general \
     --scaling-config minSize=2,maxSize=5,desiredSize=3
   ```

3. **Service Access Issues**
   ```bash
   # Check service endpoints
   kubectl get endpoints -n kubeflow
   
   # Check service configuration
   kubectl describe svc <service-name> -n kubeflow
   ```

### Performance Optimization

1. **Node Scaling**
   ```bash
   # Scale up for better performance
   aws eks update-nodegroup-config \
     --cluster-name ml-devops-cluster \
     --nodegroup-name general \
     --scaling-config minSize=3,maxSize=10,desiredSize=5
   ```

2. **Resource Limits**
   ```bash
   # Update resource requests/limits in deployments
   kubectl edit deployment xgboost-model -n kubeflow
   ```

## 🔄 Updates and Maintenance

### Update Dependencies
```bash
# Update Python packages
pip install --upgrade -r requirements.txt

# Update Kubernetes manifests
kubectl apply -f k8s/ --recursive

# Update Terraform
terraform plan
terraform apply
```

### Backup and Recovery
```bash
# Backup Terraform state
terraform state pull > terraform-state-backup.json

# Backup Kubernetes resources
kubectl get all -A -o yaml > k8s-backup.yaml

# Backup models
aws s3 sync s3://your-model-bucket ./model-backup/
```

## 🧹 Cleanup

### Remove Resources
```bash
# Delete Kubernetes resources
kubectl delete -f k8s/ --recursive

# Delete Terraform infrastructure
cd terraform
terraform destroy -auto-approve

# Clean up local files
rm -rf venv/
rm -f kubeconfig-jenkins
```

## 📚 Additional Resources

- [Kubeflow Documentation](https://www.kubeflow.org/docs/)
- [AWS EKS Documentation](https://docs.aws.amazon.com/eks/)
- [Jenkins Documentation](https://www.jenkins.io/doc/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review logs using `kubectl logs`
3. Check service status with `kubectl get`
4. Consult the main README.md file

---

**Happy Deploying! 🚀** 