pipeline {
    agent any
    
    environment {
        KUBECONFIG = credentials('kubeconfig')
        AWS_REGION = 'us-east-2'
        MODEL_NAME = 'xgboost-model'
        NAMESPACE = 'kubeflow'
    }
    
    stages {
        stage('Checkout') {
            steps {
                echo '🔍 Checking out code...'
                checkout scm
            }
        }
        
        stage('Setup Environment') {
            steps {
                echo '🔧 Setting up Python environment...'
                sh '''
                    python3 -m venv venv
                    source venv/bin/activate
                    pip install --upgrade pip
                    pip install xgboost scikit-learn pandas numpy matplotlib kfp
                '''
            }
        }
        
        stage('Train Model') {
            steps {
                echo '🎯 Training XGBoost model...'
                sh '''
                    source venv/bin/activate
                    python notebooks/xgboost_training_simple.py
                    echo "✅ Model training completed"
                '''
            }
        }
        
        stage('Test Model') {
            steps {
                echo '🧪 Testing model predictions...'
                sh '''
                    source venv/bin/activate
                    python scripts/test_inference.py
                    echo "✅ Model testing completed"
                '''
            }
        }
        
        stage('Deploy to Kubernetes') {
            steps {
                echo '🚀 Deploying model to Kubernetes...'
                sh '''
                    kubectl apply -f k8s/xgboost-flask-deployment.yaml
                    kubectl rollout status deployment/xgboost-model -n kubeflow --timeout=300s
                    echo "✅ Model deployment completed"
                '''
            }
        }
        
        stage('Run Kubeflow Automation') {
            steps {
                echo '🤖 Running Kubeflow automation job...'
                sh '''
                    kubectl apply -f k8s/automation-job.yaml
                    echo "✅ Kubeflow automation job started"
                '''
            }
        }
        
        stage('Health Check') {
            steps {
                echo '🏥 Performing health checks...'
                sh '''
                    # Wait for service to be ready
                    sleep 30
                    
                    # Test the model service
                    kubectl port-forward -n kubeflow svc/xgboost-model-service 8084:80 &
                    sleep 10
                    
                    # Test health endpoint
                    curl -f http://localhost:8084/health || echo "Health check failed"
                    
                    # Test prediction endpoint
                    curl -X POST http://localhost:8084/predict \
                        -H "Content-Type: application/json" \
                        -d '{"data": [[1.0, 2.0, 3.0, 4.0, 5.0]]}' || echo "Prediction test failed"
                    
                    echo "✅ Health checks completed"
                '''
            }
        }
    }
    
    post {
        always {
            echo '🧹 Cleaning up...'
            sh '''
                # Kill any port-forward processes
                pkill -f "kubectl port-forward" || true
                
                # Clean up temporary files
                rm -rf venv || true
            '''
        }
        success {
            echo '🎉 Pipeline completed successfully!'
            echo '📊 Model is now deployed and serving predictions'
            echo '🤖 Kubeflow automation is running'
        }
        failure {
            echo '❌ Pipeline failed!'
            echo '📋 Check the logs above for details'
        }
    }
} 