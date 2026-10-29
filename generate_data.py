import numpy as np
import pandas as pd

def generate_mempool_dataset(n_samples=50000, seed=42):
    """
    Generates a high-fidelity synthetic representation of the Flashbots Mempool Dumpster archive.
    Simulates transaction lifespans, network congestion, right-censoring, and competing risks.
    """
    print(f"Initializing data generation engine for {n_samples} samples...")
    np.random.seed(seed)
    
    # 1. Simulate network environment (Base fees follow a gamma distribution + time of day congestion)
    base_fee_gwei = np.random.gamma(shape=5, scale=8, size=n_samples) + 10 
    hour_of_day = np.random.randint(0, 24, size=n_samples)
    
    # Peak hours (13:00 to 19:00 UTC) increase network activity and push base fees higher
    congestion_mult = np.where((hour_of_day >= 13) & (hour_of_day <= 19), 1.5, 1.0)
    base_fee_gwei *= congestion_mult

    # 2. Simulate User Transaction Features
    # Priority fees (tips) follow a heavy-tailed log-normal distribution
    tip_gwei = np.random.lognormal(mean=0.2, sigma=1.2, size=n_samples) * congestion_mult
    tip_gwei = np.round(tip_gwei, 3) 
    
    # 60% standard transfers (21k gas), 20% simple swaps (65k gas), 20% complex smart contract interactions (200k gas)
    gas_limit = np.random.choice([21000, 65000, 200000], size=n_samples, p=[0.6, 0.2, 0.2])
    tip_to_base_ratio = tip_gwei / base_fee_gwei

    # 3. Model Survival Latency via Hazard Function Intuition
    # Inclusion probability increases with a higher tip ratio and decreases with high gas limits
    lambda_latent = 0.05 + 2.5 * tip_to_base_ratio - 0.002 * (gas_limit / 21000)
    lambda_latent = np.clip(lambda_latent, 0.01, 0.95) 
    
    # Calculate discrete blocks waited before inclusion using a geometric distribution
    blocks_waited = np.random.geometric(p=lambda_latent)
    
    # 4. Implement Real-World Censoring (Dropped transactions) and Competing Risks (Replacements)
    max_mempool_lifespan = 15 
    status = np.ones(n_samples, dtype=bool) # True = Successfully Included/Mined
    
    # Condition A: Right-Censoring (Low tips that stay stuck until nodes drop them entirely)
    censored_mask = (blocks_waited > max_mempool_lifespan) & (tip_gwei < 0.1)
    blocks_waited[censored_mask] = max_mempool_lifespan
    status[censored_mask] = False
    
    # Condition B: Competing Risk (User cancels/replaces the transaction with a higher tip)
    replaced_mask = (blocks_waited > 3) & (tip_gwei >= 0.1) & (tip_gwei < 1.5) & (~censored_mask)
    replaced_indices = np.where(replaced_mask)[0]
    
    # 45% of these delayed transactions get replaced/canceled by the user
    replaced_selected = np.random.choice(replaced_indices, size=int(len(replaced_indices) * 0.45), replace=False)
    blocks_waited[replaced_selected] = np.random.randint(2, 4, size=len(replaced_selected))
    status[replaced_selected] = False 

    # 5. Build and Export final structured Dataframe
    df = pd.DataFrame({
        'tip_gwei': tip_gwei,
        'base_fee_gwei': np.round(base_fee_gwei, 2),
        'tip_to_base_ratio': np.round(tip_to_base_ratio, 4),
        'gas_limit': gas_limit,
        'hour_of_day': hour_of_day,
        'blocks_waited': blocks_waited,
        'inclusion_status': status
    })
    
    return df

if __name__ == "__main__":
    dataset = generate_mempool_dataset()
    output_filename = "synthetic_mempool_dumpster.csv"
    dataset.to_csv(output_filename, index=False)
    
    print("\n✅ Step 1 Complete Successfully!")
    print(f"Data saved to file: '{output_filename}'")
    print(f"Total Rows: {dataset.shape[0]}, Total Columns: {dataset.shape[1]}")
    print("\nTransaction Status Distribution:")
    print(dataset['inclusion_status'].value_counts(normalize=True))
