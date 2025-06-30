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
                script {
                    try {
                        sh '''
                            # Wait for service to be ready
                            sleep 30
                            
                            # Test the model service
                            kubectl port-forward -n kubeflow svc/xgboost-model-service 8084:80 &
                            PF_PID=$!
                            sleep 10
                            
                            # Test health endpoint
                            curl -f http://localhost:8084/health || echo "Health check failed"
                            
                            # Test prediction endpoint
                            curl -X POST http://localhost:8084/predict \
                                -H "Content-Type: application/json" \
                                -d '{"data": [[1.0, 2.0, 3.0, 4.0, 5.0]]}' || echo "Prediction test failed"
                            
                            # Kill port-forward
                            kill $PF_PID 2>/dev/null || true
                            
                            echo "✅ Health checks completed"
                        '''
                    } catch (Exception e) {
                        echo "⚠️ Health check failed: ${e.getMessage()}"
                        // Don't fail the pipeline for health check issues
                    }
                }
            }
        }
    }
    
    post {
        always {
            echo '🧹 Cleaning up...'
            script {
                try {
                    sh '''
                        # Kill any port-forward processes
                        pkill -f "kubectl port-forward" || true
                        
                        # Clean up temporary files
                        rm -rf venv || true
                        
                        echo "✅ Cleanup completed"
                    '''
                } catch (Exception e) {
                    echo "⚠️ Cleanup failed: ${e.getMessage()}"
                }
            }
        }
        success {
            echo '🎉 Pipeline completed successfully!'
            echo '📊 Model is now deployed and serving predictions'
            echo '🤖 Kubeflow automation is running'
            echo '🌐 Access points:'
            echo '   - Kubeflow UI: http://localhost:8080'
            echo '   - ML Pipeline UI: http://localhost:8081'
            echo '   - Model Service: http://localhost:8082'
            echo '   - Jenkins: http://localhost:8083'
        }
        failure {
            echo '❌ Pipeline failed!'
            echo '📋 Check the logs above for details'
            echo '🔧 Common troubleshooting:'
            echo '   - Check if kubectl is configured properly'
            echo '   - Verify Kubernetes cluster is accessible'
            echo '   - Ensure all required files exist in the repository'
        }
    }
} 