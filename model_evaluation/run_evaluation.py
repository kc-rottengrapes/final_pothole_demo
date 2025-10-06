#!/usr/bin/env python3
"""
Complete Model Evaluation Runner
Executes all evaluation components and generates comprehensive analysis
"""

import os
import sys
from pathlib import Path

# Add parent directory to path to access model
sys.path.append(str(Path(__file__).parent.parent))

from model_evaluator import PotholeModelEvaluator
import subprocess

def install_requirements():
    """Install required packages"""
    print("📦 Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
    except subprocess.CalledProcessError:
        print("⚠️ Some packages may already be installed")

def run_complete_evaluation():
    """Run comprehensive model evaluation"""
    print("🎯 Starting Complete Model Evaluation")
    print("="*50)
    
    # Check if model exists
    model_path = "../best.pt"
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("Please ensure 'best.pt' is in the parent directory")
        return
    
    try:
        # Initialize evaluator
        evaluator = PotholeModelEvaluator(model_path)
        
        # Run evaluation
        evaluator.evaluate_model()
        
        print("\n" + "="*50)
        print("🎉 EVALUATION COMPLETED SUCCESSFULLY!")
        print("="*50)
        print("📁 Check 'evaluation_results' folder for:")
        print("   • confusion_matrix.png")
        print("   • pr_curves.png")
        print("   • confidence_histogram.png")
        print("   • metrics.json")
        print("   • evaluation_report.md")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {str(e)}")
        print("💡 Try running quick_analysis.py for basic metrics")

def run_quick_analysis():
    """Run quick analysis as backup"""
    print("\n🚀 Running Quick Analysis...")
    try:
        from quick_analysis import quick_model_analysis
        quick_model_analysis()
    except Exception as e:
        print(f"❌ Quick analysis failed: {str(e)}")

if __name__ == "__main__":
    print("🔍 POTHOLE MODEL EVALUATION SUITE")
    print("="*50)
    
    # Install requirements
    install_requirements()
    
    # Run complete evaluation
    run_complete_evaluation()
    
    # Also run quick analysis
    run_quick_analysis()
    
    print("\n✨ All evaluations complete!")
    print("Check the generated folders for detailed results.")