#!/usr/bin/env python3
"""
Quick start script for AttABseq training
"""

import subprocess
import sys
import os

def main():
    print("AttABseq Training Launcher")
    print("="*40)
    
    # Change to the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print("1. Setup environment and run training on all datasets")
    print("2. Setup environment only")
    print("3. Train specific dataset (AB645)")
    print("4. Train specific dataset (AB1101)")
    print("5. Train specific dataset (S1131)")
    print("6. Exit")
    
    choice = input("\nEnter your choice (1-6): ").strip()
    
    commands = {
        "1": [sys.executable, "setup_training.py", "--dataset", "all"],
        "2": [sys.executable, "setup_training.py", "--setup-only"],
        "3": [sys.executable, "setup_training.py", "--dataset", "AB645"],
        "4": [sys.executable, "setup_training.py", "--dataset", "AB1101"],
        "5": [sys.executable, "setup_training.py", "--dataset", "S1131"],
        "6": None
    }
    
    if choice == "6":
        print("Goodbye!")
        return 0
    
    if choice not in commands:
        print("Invalid choice!")
        return 1
    
    cmd = commands[choice]
    print(f"\nExecuting: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\n✓ Training completed successfully!")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Training failed with error code: {e.returncode}")
        return 1
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
