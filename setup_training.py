"""
AttABseq Training Setup and Configuration
This script sets up the training environment and provides utilities for running the AttABseq model.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

class AttABseqTrainer:
    def __init__(self, base_path=None):
        if base_path is None:
            self.base_path = Path(os.getcwd())
        else:
            self.base_path = Path(base_path)
        
        self.cross_validation_path = self.base_path / "cross_validation"
        self.script_path = self.cross_validation_path / "script"
        
        # Dataset configurations
        self.datasets = {
            "AB645": {
                "script": "main_AB645.py",
                "model": "model2.py",
                "data": "AB645.csv",
                "output": "output645"
            },
            "AB1101": {
                "script": "main_AB1101.py", 
                "model": "model3.py",
                "data": "AB1101.csv",
                "output": "output1101"
            },
            "S1131": {
                "script": "main_S1131.py",
                "model": "model4.py", 
                "data": "S1131.csv",
                "output": "output1131"
            }
        }
    
    def check_environment(self):
        """Check if the Python environment is properly set up."""
        print("Checking Python environment...")
        
        # Check if we're in a virtual environment
        if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            print("✓ Virtual environment detected")
        else:
            print("⚠ No virtual environment detected. Consider using a virtual environment.")
        
        # Check required packages
        required_packages = [
            'torch', 'torchvision', 'numpy', 'pandas', 
            'sklearn', 'scipy', 'seaborn', 'matplotlib', 'networkx', 'xarray'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
                print(f"✓ {package} is installed")
            except ImportError:
                missing_packages.append(package)
                print(f"✗ {package} is missing")
        
        return len(missing_packages) == 0, missing_packages
    
    def setup_directories(self):
        """Create necessary output directories."""
        print("Setting up output directories...")
        
        for dataset_name, config in self.datasets.items():
            output_base = self.cross_validation_path / config["output"]
            
            subdirs = [
                "loss_min_result", "loss_min_model",
                "best_pcc_result", "best_pcc_model", 
                "best_r2_result", "best_r2_model"
            ]
            
            for subdir in subdirs:
                dir_path = output_base / subdir
                dir_path.mkdir(parents=True, exist_ok=True)
                print(f"✓ Created {dir_path}")
    
    def setup_model_files(self):
        """Ensure model files are available with correct names."""
        print("Setting up model files...")
        
        model_mappings = [
            ("model_AB645.py", "model2.py"),
            ("model_AB1101.py", "model3.py"),
            ("model_S1131.py", "model4.py"),
            ("model_AB1101.py", "model.py")  # for predict.py
        ]
        
        for source, target in model_mappings:
            source_path = self.script_path / source
            target_path = self.script_path / target
            
            if source_path.exists() and not target_path.exists():
                shutil.copy2(source_path, target_path)
                print(f"✓ Copied {source} to {target}")
            elif target_path.exists():
                print(f"✓ {target} already exists")
            else:
                print(f"✗ Missing source file: {source}")
    
    def check_data_files(self):
        """Check if required data files exist."""
        print("Checking data files...")
        
        data_path = self.cross_validation_path / "data"
        
        for dataset_name, config in self.datasets.items():
            data_file = data_path / config["data"]
            if data_file.exists():
                print(f"✓ {config['data']} found")
            else:
                print(f"✗ {config['data']} missing")
        
        # Check if BLAST database exists
        blast_path = self.cross_validation_path / "ncbi-blast-2.12.0+"
        if blast_path.exists():
            print("✓ BLAST database directory found")
        else:
            print("✗ BLAST database directory missing")
            print("  Note: BLAST is required for PSSM generation but may work without it")
    
    def run_training(self, dataset=None, dry_run=False):
        """Run training for specified dataset(s)."""
        if dataset and dataset not in self.datasets:
            print(f"Error: Unknown dataset '{dataset}'. Available: {list(self.datasets.keys())}")
            return False
        
        datasets_to_run = [dataset] if dataset else list(self.datasets.keys())
        
        for dataset_name in datasets_to_run:
            config = self.datasets[dataset_name]
            script_file = self.script_path / config["script"]
            
            if not script_file.exists():
                print(f"✗ Training script missing: {script_file}")
                continue
            
            print(f"\n{'='*50}")
            print(f"Training dataset: {dataset_name}")
            print(f"Script: {script_file}")
            print(f"{'='*50}")
            
            if dry_run:
                print(f"DRY RUN: Would execute {script_file}")
                continue
            
            # Change to script directory and run
            original_dir = os.getcwd()
            try:
                os.chdir(self.script_path)
                
                # Use the configured Python executable
                python_exe = sys.executable
                result = subprocess.run([python_exe, config["script"]], 
                                      capture_output=False, text=True)
                
                if result.returncode == 0:
                    print(f"✓ Training completed for {dataset_name}")
                else:
                    print(f"✗ Training failed for {dataset_name}")
                    return False
                    
            except Exception as e:
                print(f"✗ Error running training for {dataset_name}: {e}")
                return False
            finally:
                os.chdir(original_dir)
        
        return True
    
    def full_setup(self):
        """Run complete setup process."""
        print("AttABseq Training Setup")
        print("="*50)
        
        # Check environment
        env_ok, missing = self.check_environment()
        if not env_ok:
            print(f"Missing packages: {missing}")
            print("Install them with: pip install -r requirements.txt")
            return False
        
        # Setup directories
        self.setup_directories()
        
        # Setup model files
        self.setup_model_files()
        
        # Check data
        self.check_data_files()
        
        print("\n✓ Setup completed successfully!")
        print("\nTo run training:")
        print("  python train.py --dataset AB1101    # Train on AB1101 dataset")
        print("  python train.py --dataset all       # Train on all datasets")
        print("  python train.py --setup-only        # Setup only, no training")
        
        return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="AttABseq Training Setup and Runner")
    parser.add_argument("--dataset", choices=["AB645", "AB1101", "S1131", "all"], 
                       help="Dataset to train on")
    parser.add_argument("--setup-only", action="store_true", 
                       help="Only run setup, don't train")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be done without actually doing it")
    
    args = parser.parse_args()
    
    trainer = AttABseqTrainer()
    
    # Always run setup first
    if not trainer.full_setup():
        return 1
    
    # Run training if requested
    if not args.setup_only:
        dataset = args.dataset if args.dataset != "all" else None
        if not trainer.run_training(dataset, dry_run=args.dry_run):
            return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
