import argparse
import subprocess
import sys
import os

# Define paths to each script
SCRIPTS = {
    "feature_prediction": "a.py",
    "promo_prediction": "b.py",
    "merge_actual": "c.py",
    "merge_predicted": "d.py",
    "feature_engineering": "e.py",
    "model_training": "f.py",
    "dashboard": "g.py",
}

def run_script(script_name):
    print(f"▶ Running: {script_name}")
    subprocess.run([sys.executable, SCRIPTS[script_name]], check=True)

def run_processing_pipeline():
    run_script("feature_prediction")
    run_script("promo_prediction")
    run_script("merge_actual")
    run_script("merge_predicted")
    run_script("feature_engineering")
    run_script("model_training")

def run_dashboard():
    print("▶ Launching Streamlit Dashboard...")
    subprocess.run(["streamlit", "run", SCRIPTS["dashboard"]], check=True)

def main():
    parser = argparse.ArgumentParser(description="Tyre Sales Forecasting Pipeline")
    parser.add_argument("--mode", choices=["dashboard", "process", "full"], required=True,
                        help="Select pipeline mode: 'dashboard', 'process', or 'full'")

    args = parser.parse_args()

    if args.mode == "dashboard":
        run_dashboard()
    elif args.mode == "process":
        run_processing_pipeline()
    elif args.mode == "full":
        run_processing_pipeline()
        run_dashboard()

if __name__ == "__main__":
    main()
