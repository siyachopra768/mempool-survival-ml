import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest

def run_ml_pipeline():
    print("🚀 Ingesting synthetic_mempool_dumpster.csv...")
    try:
        df = pd.read_csv("synthetic_mempool_dumpster.csv")
    except FileNotFoundError:
        print("❌ Error: Dataset file not found! Please run generate_data.py first.")
        return

    # 1. Isolate the feature space (X)
    feature_cols = ['tip_gwei', 'base_fee_gwei', 'tip_to_base_ratio', 'gas_limit', 'hour_of_day']
    X = df[feature_cols]

    # 2. Format targets into a Structured Array (Strictly required by scikit-survival)
    # Binary status (Event Indicator) + numeric blocks_waited (Time-to-Event Duration)
    y = np.array(
        list(zip(df['inclusion_status'], df['blocks_waited'])),
        dtype=[('Status', '?'), ('Survival_in_blocks', '<f8')]
    )

    # 3. Partition Data into Train and Test splits (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    print(f"📦 Data partition complete. Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # 4. Initialize and Train the Gradient Boosting Survival Model
    print("\n🧠 Training Gradient Boosting Hazard Model (Estimating non-linear latency curves)...")
    model = RandomSurvivalForest(
        n_estimators=100,
        min_samples_leaf=5,
        n_jobs=-1,
        min_samples_split=10,
        random_state=42
    )
    model.fit(X_train, y_train)
    print("✅ Model training finished successfully!")

    # 5. Model Performance Evaluation via Concordance Index (C-Index)
    # The C-Index evaluates ranking accuracy (0.5 = random guess, 1.0 = absolute prediction perfection)
    c_index = model.score(X_test, y_test)
    print(f"\n========================================================")
    print(f"📈 Model Evaluation: Concordance Index = {c_index:.4f}")
    print(f"========================================================")
    
    # 6. Extract Predictive Visual Assets for Portfolio Representation
    print("\n📊 Extrapolating step-down transaction survival curves for GitHub...")
    # Select 3 specific test transactions with different tips to inspect behavior
    sample_indices = [0, 10, 50]
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
    
    # Export chart directly into directory
    plt.savefig("mempool_survival_curves.png", dpi=300)  
    print("💾 Graphical asset exported as 'mempool_survival_curves.png'!")

if __name__ == "__main__":
    run_ml_pipeline()
