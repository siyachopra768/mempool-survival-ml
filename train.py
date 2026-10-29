import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest
import joblib
# Configure clean logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def parse_args():
    parser = argparse.ArgumentParser(description="Ethereum Mempool Transaction Survival Analysis Pipeline")
    parser.add_argument("--data", type=str, default="synthetic_mempool_dumpster.csv", help="Path to dataset CSV")
    parser.add_argument("--sample-size", type=int, default=15000, help="Subsample rows for lightning-fast training")
    parser.add_argument("--estimators", type=int, default=80, help="Number of trees in the forest")
    return parser.parse_args()

def run_ml_pipeline():
    args = parse_args()
    
    logging.info(f"Ingesting dataset from {args.data}...")
    try:
        df = pd.read_csv(args.data)
    except FileNotFoundError:
        logging.error("Dataset file not found! Please ensure 'synthetic_mempool_dumpster.csv' is generated.")
        return

    # Subsample dataset to optimize memory and training speed without losing distribution integrity
    if len(df) > args.sample_size:
        logging.info(f"Subsampling dataset from {len(df)} to {args.sample_size} rows for rapid iteration...")
        df = df.sample(n=args.sample_size, random_state=42).reset_index(drop=True)

    # 1. Advanced Feature Engineering
    logging.info("Engineering feature space...")
    df['tip_dominance'] = df['tip_gwei'] / (df['base_fee_gwei'] + 1e-8)
    
    feature_cols = [
        'tip_gwei', 
        'base_fee_gwei', 
        'tip_to_base_ratio', 
        'tip_dominance', 
        'gas_limit', 
        'hour_of_day'
    ]
    X = df[feature_cols]

    # 2. Format targets into a Structured Array (Required by scikit-survival)
    y = np.array(
        list(zip(df['inclusion_status'], df['blocks_waited'])),
        dtype=[('Status', '?'), ('Survival_in_blocks', '<f8')]
    )

    # 3. Partition Data (80/20 train-test split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    logging.info(f"Data partition complete. Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # 4. Initialize and Train with Balanced Parameters
    logging.info("Training Random Survival Forest (optimized for speed & predictive power)...")
    model = RandomSurvivalForest(
        n_estimators=args.estimators,
        max_depth=15,          # Allows enough depth to capture non-linear survival hazards
        min_samples_split=10,
        min_samples_leaf=5,
        n_jobs=-1,             # Utilize all available CPU cores
        random_state=42
    )
    model.fit(X_train, y_train)
    model_path = "mempool_rsf_model.pkl"
    joblib.dump(model,model_path)
    logging.info(f"Trained model serialized and saved to '{model_path}'")
    logging.info("Model training finished successfully!")

    # 5. Model Evaluation
    c_index = model.score(X_test, y_test)
    logging.info(f"Model Evaluation (C-Index): {c_index:.4f}")

    # 6. Extract Predictive Visual Assets
    logging.info("Generating transaction survival curves for portfolio...")
    sample_indices = [0, 5, 10]
    sample_X = X_test.iloc[sample_indices]
    survival_functions = model.predict_survival_function(sample_X)

    plt.figure(figsize=(10, 6))
    for idx, fn in enumerate(survival_functions):
        actual_tip = sample_X.iloc[idx]['tip_gwei']
        plt.step(fn.x, fn(fn.x), where="post", label=f"Tx Profile {idx+1} (Tip: {actual_tip:.3f} Gwei)")

    plt.title("Ethereum Mempool Transaction Survival Probability Curves")
    plt.xlabel("Time Elapsed (Blocks)")
    plt.ylabel("Probability of Remaining Trapped in Mempool Queue")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    
    plt.savefig("mempool_survival_curves.png", dpi=300)
    logging.info("Graphical asset exported successfully as 'mempool_survival_curves.png'.")

if __name__ == "__main__":
    run_ml_pipeline()