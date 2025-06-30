# MLDevops: XGBoost Model with Kubeflow & Jenkins Automation

A complete Machine Learning DevOps pipeline demonstrating automated XGBoost model training, deployment, and serving using AWS EKS, Kubeflow, and Jenkins.

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Jenkins CI/CD │───▶│  Kubeflow ML    │───▶│  Model Serving  │
│   Pipeline      │    │  Platform       │    │  (Flask API)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   AWS EKS       │    │   Kubernetes    │    │   S3 Storage    │
│   Infrastructure│    │   Orchestration │    │   (Models)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- AWS CLI configured
- kubectl installed
- Terraform installed
- Python 3.9+

### 1. Infrastructure Setup
```bash
# Initialize Terraform
cd terraform
terraform init
terraform plan
terraform apply
```

### 2. Kubeflow Installation
```bash
# Install Kubeflow components
./scripts/install_kubeflow.sh
```

### 3. Model Training
```bash
# Train XGBoost model
python notebooks/xgboost_training_simple.py
```

### 4. Deploy Model
```bash
# Deploy to Kubernetes
kubectl apply -f k8s/xgboost-flask-deployment.yaml
```

### 5. Access Services
```bash
# Port-forward services
kubectl port-forward -n kubeflow svc/centraldashboard 8080:80 &
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8081:80 &
kubectl port-forward -n kubeflow svc/xgboost-model-service 8082:80 &
kubectl port-forward -n jenkins svc/jenkins 8083:80 &
```

## 📊 Services & URLs

| Service | URL | Description |
|---------|-----|-------------|
| Kubeflow Dashboard | http://localhost:8080 | Main Kubeflow UI |
| ML Pipeline UI | http://localhost:8081 | Kubeflow Pipelines |
| Model API | http://localhost:8082 | XGBoost Model Service |
| Jenkins | http://localhost:8083/jenkins/ | CI/CD Pipeline |

## 🤖 Automation Features

### Jenkins Pipeline
- **Automated Model Training**: Triggers on code changes
- **Kubernetes Deployment**: Automatic model deployment
- **Health Checks**: Validates model service
- **Integration**: Connects with Kubeflow automation

### Kubeflow Automation
- **Scheduled Jobs**: Runs every 30 minutes
- **Data Generation**: Synthetic data creation
- **Model Training**: Automated XGBoost training
- **Evaluation**: Model performance metrics
- **Deployment**: Kubernetes manifest generation

## 📁 Project Structure

```
xgboostOps/
├── terraform/                 # AWS Infrastructure
│   ├── main.tf               # EKS, S3, RDS configuration
│   └── outputs.tf            # Infrastructure outputs
├── k8s/                      # Kubernetes Manifests
│   ├── xgboost-flask-deployment.yaml
│   ├── automation-job.yaml
│   ├── scheduled-automation.yaml
│   └── jenkins-deployment.yaml
├── notebooks/                # Jupyter Notebooks
│   └── xgboost_training_simple.py
├── models/                   # Model Files
│   ├── flask_model_server.py
│   └── Dockerfile
├── scripts/                  # Automation Scripts
│   ├── deploy.sh
│   ├── install_kubeflow.sh
│   └── test_inference.py
├── pipelines/                # Kubeflow Pipelines
│   ├── xgboost_pipeline.py
│   └── simple_pipeline.py
├── Jenkinsfile              # Jenkins CI/CD Pipeline
└── README.md                # This file
```

## 🔧 Configuration

### Environment Variables
```bash
export AWS_REGION=us-east-2
export AWS_ACCOUNT_ID=your-account-id
export KUBECONFIG=~/.kube/config
```

### Model Configuration
- **Algorithm**: XGBoost Regressor
- **Features**: 5 synthetic features
- **Training Samples**: 1000
- **Test Split**: 80/20
- **Hyperparameters**: 
  - n_estimators: 100
  - max_depth: 6
  - learning_rate: 0.1

## 📈 Model Performance

The XGBoost model typically achieves:
- **R² Score**: ~0.85-0.95
- **RMSE**: ~0.1-0.2
- **Training Time**: <30 seconds
- **Prediction Latency**: <10ms

## 🧪 Testing

### Model Testing
```bash
python scripts/test_inference.py
```

### API Testing
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

## 🔄 CI/CD Pipeline

### Jenkins Pipeline Stages
1. **Checkout**: Git repository checkout
2. **Setup**: Python environment setup
3. **Train**: XGBoost model training
4. **Test**: Model validation
5. **Deploy**: Kubernetes deployment
6. **Automation**: Kubeflow job trigger
7. **Health Check**: Service validation

### Triggering Pipeline
- **Manual**: Build Now in Jenkins
- **Git Hook**: On code push
- **Schedule**: Cron-based triggers

## 🛠️ Troubleshooting

### Common Issues

1. **PVC Binding Issues**
   ```bash
   kubectl get pvc -n kubeflow
   kubectl describe pvc <pvc-name>
   ```

2. **Pod Resource Issues**
   ```bash
   kubectl describe pod <pod-name> -n kubeflow
   kubectl top nodes
   ```

3. **Service Access Issues**
   ```bash
   kubectl get svc -n kubeflow
   kubectl port-forward <service-name> <local-port>:<service-port>
   ```

### Logs
```bash
# Jenkins logs
kubectl logs -n jenkins <jenkins-pod-name>

# Model service logs
kubectl logs -n kubeflow deployment/xgboost-model

# Kubeflow automation logs
kubectl logs -n kubeflow job/xgboost-automation-job
```

## 📚 Next Steps

1. **Production Deployment**
   - Set up proper ingress/load balancer
   - Configure monitoring and alerting
   - Implement proper secrets management

2. **Advanced Features**
   - Model versioning with MLflow
   - A/B testing with Seldon Core
   - Automated hyperparameter tuning
   - Data pipeline integration

3. **Monitoring & Observability**
   - Prometheus metrics
   - Grafana dashboards
   - Distributed tracing
   - Model drift detection

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Kubeflow community
- XGBoost developers
- Jenkins community
- AWS EKS team

---

**Happy ML DevOps! 🚀** 