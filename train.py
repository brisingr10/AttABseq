#!/usr/bin/env python3
"""
AttABseq Training Runner
A unified script to train the AttABseq model on different datasets.
"""

import os
import sys
import argparse
import subprocess

def setup_environment():
    """Setup the Python environment and install dependencies."""
    print("Setting up Python environment...")
    
    # Check if requirements.txt exists
    if not os.path.exists("requirements.txt"):
        print("Error: requirements.txt not found!")
        return False
    
    # Install requirements
    try:
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], check=True, capture_output=True, text=True)
        print("Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False

def run_training(dataset):
    """Run training for a specific dataset."""
    script_map = {
        "AB645": "main_AB645.py",
        "AB1101": "main_AB1101.py", 
        "S1131": "main_S1131.py"
    }
    
    if dataset not in script_map:
        print(f"Error: Dataset {dataset} not supported. Choose from: {list(script_map.keys())}")
        return False
    
    script_path = os.path.join("cross_validation", "script", script_map[dataset])
    
    if not os.path.exists(script_path):
        print(f"Error: Training script {script_path} not found!")
        return False
    
    print(f"Starting training for dataset {dataset}...")
    print(f"Running script: {script_path}")
    
    # Change to the script directory
    original_dir = os.getcwd()
    script_dir = os.path.join("cross_validation", "script")
    
    try:
        os.chdir(script_dir)
        result = subprocess.run([sys.executable, script_map[dataset]], 
                              check=True, capture_output=False, text=True)
        print(f"Training completed successfully for {dataset}!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error during training: {e}")
        return False
    finally:
        os.chdir(original_dir)

def main():
    parser = argparse.ArgumentParser(description="AttABseq Training Runner")
    parser.add_argument("--dataset", choices=["AB645", "AB1101", "S1131", "all"], 
                       default="all", help="Dataset to train on (default: all)")
    parser.add_argument("--setup-only", action="store_true", 
                       help="Only setup environment, don't run training")
    
    args = parser.parse_args()
    
    # Setup environment
    if not setup_environment():
        print("Failed to setup environment!")
        return 1
    
    if args.setup_only:
        print("Environment setup completed!")
        return 0
    
    # Run training
    if args.dataset == "all":
        datasets = ["AB645", "AB1101", "S1131"]
        for dataset in datasets:
            print(f"\n{'='*50}")
            print(f"Training dataset: {dataset}")
            print(f"{'='*50}")
            if not run_training(dataset):
                print(f"Training failed for {dataset}")
                return 1
    else:
        if not run_training(args.dataset):
            print(f"Training failed for {args.dataset}")
            return 1
    
    print("\nAll training completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
