 Ethereum Mempool Survival Analysis & Real-Time Inclusion Engine

A production-grade, asynchronous machine learning pipeline that predicts the probability of Ethereum transaction inclusion across sequential blocks using **Survival Analysis (Random Survival Forests)** and live WebSocket streaming.


 The Problem & Solution

In Ethereum's gas market (EIP-1559), standard classification models fail because transaction inclusion is fundamentally a **time-to-event** problem. Knowing *if* a transaction clears isn't enough—traders and MEV bots need to know **how many blocks** a transaction will wait before inclusion, especially during network congestion.

This system solves the problem through a two-tier architecture:
1. **Offline Training Pipeline:** Trains a Random Survival Forest model on historical/synthetic gas distributions to map fee structures to survival hazard curves.
2. **Real-Time Streaming Daemon:** Connects via asynchronous WebSockets (`AsyncWeb3`) to ingest live mainnet pending transactions, engineer real-time features, and calculate live inclusion hazard probabilities on the fly.

System Architecture


[ Ethereum Mainnet WSS ] 
         │
         ▼
[ Async WebSocket Daemon ] ──► [ Feature Engineering ] ──► [ Pre-trained RSF Model ] ──► [ Live Inclusion Probabilities ]
         ▲
         │ (joblib)
[ Offline Training Pipeline ]


Tech Stack
Language: Python 3.10+

Blockchain Interaction: web3.py (AsyncWeb3, WebSocketProvider)

Machine Learning: scikit-survival (Random Survival Forest), scikit-learn, pandas, numpy, joblib

Concurrency: Python asyncio for non-blocking event loops





Getting Started
1. Clone & Setup Environment
Bash
git clone [https://github.com/your-username/mempool-survival-ml.git](https://github.com/siyachopra768/mempool-survival-ml.git)
cd mempool-survival-ml
conda create -n mempool-env python=3.10
conda activate mempool-env
pip install -r requirements.txt
2. Train the Survival Model
Run the offline training script to generate and serialize the Random Survival Forest model:

Bash
python train.py
(This outputs mempool_rsf_model.pkl and your model evaluation plots).

3. Run the Real-Time Ingestion Daemon
Configure your Ethereum node WebSocket provider URL (e.g., Alchemy or Infura) in stream_inference.py, then start the live scoring daemon:

Bash
python stream_inference.py


