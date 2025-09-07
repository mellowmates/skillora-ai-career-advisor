#!/usr/bin/env python3
"""
Master script to run the complete ML pipeline:
1. Preprocess raw data
2. Train all models
3. Save trained models
"""

import os
import sys
import subprocess
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(command, description):
    """Run a command and handle errors"""
    logger.info(f"🚀 {description}")
    
    # Use virtual environment python if available
    if os.path.exists('venv/bin/python'):
        python_cmd = 'venv/bin/python'
    elif os.path.exists('venv/Scripts/python.exe'):
        python_cmd = 'venv/Scripts/python.exe'
    else:
        python_cmd = 'python3'
    
    # Replace python3 with venv python
    command = command.replace('python3', python_cmd)
    logger.info(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        logger.info(f"✅ {description} completed successfully")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} failed")
        logger.error(f"Error: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False

def check_directories():
    """Ensure all required directories exist"""
    directories = [
        'data/processed',
        'models/trained_models',
        'models/training'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"📁 Directory ensured: {directory}")

def main():
    """Run the complete ML pipeline"""
    start_time = datetime.now()
    logger.info("🎯 Starting Skillora AI Model Training Pipeline")
    logger.info(f"Start time: {start_time}")
    
    # Check directories
    check_directories()
    
    # Step 1: Preprocess data
    logger.info("\n" + "="*50)
    logger.info("STEP 1: DATA PREPROCESSING")
    logger.info("="*50)
    
    if not run_command("python3 models/data_collection/data_preprocessor.py", "Data preprocessing"):
        logger.error("Data preprocessing failed. Stopping pipeline.")
        return False
    
    # Step 2: Train Career Model
    logger.info("\n" + "="*50)
    logger.info("STEP 2: CAREER MODEL TRAINING")
    logger.info("="*50)
    
    if not run_command("python3 models/training/train_career_model.py", "Career model training"):
        logger.error("Career model training failed.")
        return False
    
    # Step 3: Train Salary Model
    logger.info("\n" + "="*50)
    logger.info("STEP 3: SALARY MODEL TRAINING")
    logger.info("="*50)
    
    if not run_command("python3 models/training/train_salary_model.py", "Salary model training"):
        logger.error("Salary model training failed.")
        return False
    
    # Step 4: Train Skills Model
    logger.info("\n" + "="*50)
    logger.info("STEP 4: SKILLS MODEL TRAINING")
    logger.info("="*50)
    
    if not run_command("python3 models/training/train_skills_model.py", "Skills model training"):
        logger.error("Skills model training failed.")
        return False
    
    # Summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    logger.info("\n" + "="*50)
    logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("="*50)
    logger.info(f"Total duration: {duration}")
    logger.info(f"End time: {end_time}")
    
    # List generated files
    logger.info("\n📋 Generated Files:")
    
    processed_files = [
        'data/processed/preprocessed_data.csv',
        'data/processed/feature_info.json',
        'data/processed/encoders.pkl'
    ]
    
    model_files = [
        'models/trained_models/career_model.pkl',
        'models/trained_models/career_model_metadata.json',
        'models/trained_models/salary_model.pkl',
        'models/trained_models/salary_model_metadata.json',
        'models/trained_models/skills_model.pkl',
        'models/trained_models/skills_model_metadata.json'
    ]
    
    all_files = processed_files + model_files
    
    for file_path in all_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            logger.info(f"  ✅ {file_path} ({size:,} bytes)")
        else:
            logger.warning(f"  ❌ {file_path} (missing)")
    
    logger.info("\n🚀 Your models are ready to use!")
    logger.info("You can now run your Flask application with trained models.")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)