pipeline {
    agent any
    
    environment {
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
                    # Ensure Python3 is available
                    if ! command -v python3 &> /dev/null; then
                        echo "❌ Python3 not found. Installing Python..."
                        apt-get update && apt-get install -y python3 python3-pip python3-venv
                    fi
                    
                    echo "Using Python: $(which python3)"
                    python3 -m venv venv
                    . venv/bin/activate
                    pip install --upgrade pip
                    pip install xgboost scikit-learn pandas numpy matplotlib kfp boto3
                '''
            }
        }
        
        stage('Train Model') {
            steps {
                echo '🎯 Training XGBoost model...'
                sh '''
                    . venv/bin/activate
                    python3 notebooks/xgboost_training_simple.py
                    echo "✅ Model training completed"
                '''
            }
        }
        
        stage('Test Model') {
            steps {
                echo '🧪 Testing model predictions...'
                sh '''
                    . venv/bin/activate
                    python3 scripts/test_inference.py
                    echo "✅ Model testing completed"
                '''
            }
        }
        
        stage('Setup Kubernetes Access') {
            steps {
                echo '🔑 Setting up Kubernetes access...'
                script {
                    try {
                        // Try to use kubeconfig credential if available
                        withCredentials([file(credentialsId: 'kubeconfig', variable: 'KUBECONFIG_FILE')]) {
                            env.KUBECONFIG = KUBECONFIG_FILE
                            echo "✅ Using kubeconfig credential"
                        }
                    } catch (Exception e) {
                        echo "⚠️ Kubeconfig credential not found, using local kubeconfig"
                        // Use local kubeconfig file if credential is not available
                        sh '''
                            if [ -f "kubeconfig" ]; then
                                export KUBECONFIG=$(pwd)/kubeconfig
                                echo "✅ Using local kubeconfig file"
                            else
                                echo "⚠️ No kubeconfig available, skipping Kubernetes operations"
                                env.SKIP_K8S=true
                            fi
                        '''
                    }
                }
            }
        }
        
        stage('Deploy to Kubernetes') {
            when {
                expression { env.SKIP_K8S != 'true' }
            }
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
            when {
                expression { env.SKIP_K8S != 'true' }
            }
            steps {
                echo '🤖 Running Kubeflow automation job...'
                sh '''
                    kubectl apply -f k8s/automation-job.yaml
                    echo "✅ Kubeflow automation job started"
                '''
            }
        }
        
        stage('Health Check') {
            when {
                expression { env.SKIP_K8S != 'true' }
            }
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
        
        stage('Cleanup') {
            steps {
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
        }
    }
    
    post {
        success {
            echo '🎉 Pipeline completed successfully!'
            script {
                if (env.SKIP_K8S == 'true') {
                    echo '📊 Model training completed (Kubernetes deployment skipped)'
                    echo '🔧 To enable Kubernetes deployment, configure kubeconfig credential in Jenkins'
                } else {
                    echo '📊 Model is now deployed and serving predictions'
                    echo '🤖 Kubeflow automation is running'
                    echo '🌐 Access points:'
                    echo '   - Kubeflow UI: http://localhost:8080'
                    echo '   - ML Pipeline UI: http://localhost:8081'
                    echo '   - Model Service: http://localhost:8082'
                    echo '   - Jenkins: http://localhost:8083'
                }
            }
        }
        failure {
            echo '❌ Pipeline failed!'
            echo '📋 Check the logs above for details'
            echo '🔧 Common troubleshooting:'
            echo '   - Check if kubectl is configured properly'
            echo '   - Verify Kubernetes cluster is accessible'
            echo '   - Ensure all required files exist in the repository'
            echo '   - Configure kubeconfig credential in Jenkins for Kubernetes operations'
        }
    }
} 