#!/usr/bin/env python3
"""
Simple script to run the XGBoost Kubeflow Pipeline
"""

import os
import sys
import kfp
from kfp import compiler
import subprocess

def main():
    """Main function to run the simple pipeline"""
    
    # Add the pipelines directory to Python path
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'pipelines'))
    
    # Import the pipeline
    from simple_pipeline import simple_xgboost_pipeline
    
    # Compile the pipeline
    print("🔧 Compiling simple pipeline...")
    compiler.Compiler().compile(simple_xgboost_pipeline, 'simple_xgboost_pipeline.yaml')
    print("✅ Pipeline compiled successfully!")
    
    # Get the Kubeflow API endpoint
    print("🌐 Getting Kubeflow API endpoint...")
    try:
        # Get the ML Pipeline service
        result = subprocess.run(
            ['kubectl', 'get', 'svc', 'ml-pipeline', '-n', 'kubeflow', '-o', 'jsonpath={.spec.clusterIP}'],
            capture_output=True, text=True, check=True
        )
        cluster_ip = result.stdout.strip()
        
        if cluster_ip:
            api_endpoint = f"http://{cluster_ip}:8888"
            print(f"✅ Found ML Pipeline API at: {api_endpoint}")
        else:
            print("❌ Could not find ML Pipeline service")
            return
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Error getting ML Pipeline service: {e}")
        return
    
    # Create the client
    print("🔗 Creating Kubeflow client...")
    client = kfp.Client(host=api_endpoint)
    
    # Upload and run the pipeline
    print("📤 Uploading pipeline...")
    try:
        pipeline = client.upload_pipeline(
            pipeline_package_path='simple_xgboost_pipeline.yaml',
            pipeline_name='Simple XGBoost Pipeline'
        )
        print(f"✅ Pipeline uploaded with ID: {pipeline.id}")
        
        # Run the pipeline
        print("🚀 Starting pipeline run...")
        run = client.create_run_from_pipeline_func(
            simple_xgboost_pipeline,
            arguments={},
            run_name='Simple XGBoost Training Run'
        )
        print(f"✅ Pipeline run started with ID: {run.run_id}")
        
        # Get the run URL
        run_url = f"http://localhost:8081/#/runs/details/{run.run_id}"
        print(f"🌐 View pipeline run at: {run_url}")
        print("📊 You can now see the automation in the Kubeflow UI!")
        
    except Exception as e:
        print(f"❌ Error running pipeline: {e}")
        print("💡 Make sure the ML Pipeline UI is accessible at http://localhost:8081")

if __name__ == '__main__':
    main() 