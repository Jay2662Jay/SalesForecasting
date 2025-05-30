import argparse
import subprocess
import sys
import os

# Define paths to each script
SCRIPTS = {
    "feature_prediction": "1_feature_prediction.py",
    "promo_prediction": "1_add_promo_prediction.py",
    "merge_actual": "2_merging_actual_data.py",
    "merge_predicted": "2_merging_predicted_data.py",
    "feature_engineering": "3_1_feature_engineering.py",
    "model_training": "5_3_catboost_optuna.py",
    "dashboard": "6_1_dashboard_streamlit.py",
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
