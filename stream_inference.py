import asyncio
import logging
import joblib
import pandas as pd
from web3 import AsyncWeb3, WebSocketProvider

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Load the pre-trained Random Survival Forest model
MODEL_PATH = "mempool_rsf_model.pkl"
try:
    model = joblib.load(MODEL_PATH)
    logging.info("✅ Loaded pre-trained survival model successfully.")
except FileNotFoundError:
    logging.error(f"❌ Model file '{MODEL_PATH}' not found! Run 'python trial.py' first.")
    exit(1)

# Replace with your actual Alchemy or Infura WebSocket URL
WSS_URL = "wss://eth-mainnet.g.alchemy.com/v2/alch_GNnU5LUGQ1KaZNTHiMl7v"

async def process_pending_transactions():
    logging.info("🔌 Connecting to Ethereum WebSocket feed for real-time mempool ingestion...")
    
    # Initialize WebSocketProvider cleanly
    provider = WebSocketProvider(WSS_URL)

    async with AsyncWeb3(provider) as w3:
        subscription_id = await w3.eth.subscribe("newPendingTransactions")
        logging.info(f"📡 Subscribed successfully! Subscription ID: {subscription_id}")

        counter = 0
        async for response in w3.socket.process_subscriptions():
            try:
                tx_hash = response.get("result")
                if not tx_hash:
                    continue

                # Throttle processing: sample every 5th transaction to keep CPU light and smooth
                counter += 1
                if counter % 5 != 0:
                    continue

                # Fetch full transaction details asynchronously
                tx = await w3.eth.get_transaction(tx_hash)
                if not tx:
                    continue

                # Extract raw transaction fields
                gas_limit = tx.get("gas", 21000)
                max_fee_gwei = w3.from_wei(tx.get("maxFeePerGas", tx.get("gasPrice", 0)), "gwei")
                max_priority_fee_gwei = w3.from_wei(tx.get("maxPriorityFeePerGas", 0), "gwei")
                
                # Fetch latest block base fee
                latest_block = await w3.eth.get_block("latest")
                base_fee_gwei = w3.from_wei(latest_block.get("baseFeePerGas", 15 * 10**9), "gwei")

                tip_gwei = float(max_priority_fee_gwei) if max_priority_fee_gwei > 0 else float(max_fee_gwei - base_fee_gwei)
                tip_gwei = max(0.0, tip_gwei)

                # Feature Engineering matching training pipeline
                tip_to_base_ratio = tip_gwei / (float(base_fee_gwei) + 1e-8)
                tip_dominance = tip_gwei / (float(base_fee_gwei) + 1e-8)
                hour_of_day = pd.Timestamp.now().hour

                # Construct feature vector DataFrame
                live_features = pd.DataFrame([{
                    'tip_gwei': tip_gwei,
                    'base_fee_gwei': float(base_fee_gwei),
                    'tip_to_base_ratio': tip_to_base_ratio,
                    'tip_dominance': tip_dominance,
                    'gas_limit': int(gas_limit),
                    'hour_of_day': hour_of_day
                }])

                # Real-Time Inference
                survival_funcs = model.predict_survival_function(live_features)
                prob_trapped_block_1 = survival_funcs[0](1.0)

                logging.info(f"🔍 [Live Tx] Hash: {tx_hash.hex()[:10]}... | Tip: {tip_gwei:.2f} Gwei | P(Trapped in Blk 1): {prob_trapped_block_1:.2f}")

            except Exception as e:
                # Gracefully skip transient dropouts or malformed payloads
                continue

async def main():
    while True:
        try:
            await process_pending_transactions()
        except Exception as e:
            logging.warning(f"⚠️ Connection dropped ({e}). Reconnecting in 5 seconds...")
            await asyncio.sleep(5)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("🛑 Real-time mempool ingestion daemon stopped by user.")