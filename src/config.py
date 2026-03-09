"""
Fed-Intel Configuration
=======================
Central configuration for the entire Fed-Intel system.
All paths, API keys, model parameters, and hyperparameters are defined here.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─── Project Paths ───────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATASET_IDS2018_DIR = DATA_DIR / "nf-cse-cic-ids2018-v3"
DATASET_GOTHAM_DIR = DATA_DIR / "gotham-2025"
MODELS_DIR = PROJECT_ROOT / "models"
CHROMA_DIR = PROJECT_ROOT / "chromadb_store"

# ─── API Keys ────────────────────────────────────────────────────────────────

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# ─── LLM Configuration ──────────────────────────────────────────────────────

LLM_PRIMARY_MODEL = "gemini-2.0-flash"
LLM_FALLBACK_MODEL = "llama-3.1-70b-versatile"  # Groq
LLM_TEMPERATURE = 0.1  # Low temp for deterministic security analysis
LLM_MAX_RETRIES = 3

# ─── Federated Learning ─────────────────────────────────────────────────────

FL_NUM_ROUNDS = 10          # Number of federated training rounds
FL_NUM_CLIENTS = 3          # Number of company nodes (A, B, C)
FL_LOCAL_EPOCHS = 5         # Local training epochs per FL round
FL_BATCH_SIZE = 64
FL_LEARNING_RATE = 1e-3
FL_TRUST_EPSILON = 1e-6     # ε in τ_i = 1/(L_i + ε) to avoid division by zero

# ─── IDS Model ───────────────────────────────────────────────────────────────

IDS_HIDDEN_LAYERS = [128, 64, 32]   # MLP hidden layer sizes
IDS_DROPOUT = 0.3
IDS_INPUT_DIM = 53                   # NF-CSE-CIC-IDS2018-v3 has 53 features
IDS_GOTHAM_INPUT_DIM = 23            # Gotham 2025 has 23 features
IDS_COMMON_DIM = 23                  # Aligned feature space (shared subset)

# ─── Differential Privacy ───────────────────────────────────────────────────

DP_EPSILON = 1.0            # Privacy budget (lower = more private)
DP_NOISE_SIGMA = 0.1        # Gaussian noise std (derived from ε)

# ─── Embeddings ──────────────────────────────────────────────────────────────

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# ─── ChromaDB Collections ───────────────────────────────────────────────────

CHROMA_COLLECTION_THREATS = "threat_summaries"
CHROMA_COLLECTION_DEFENSE = "defense_patterns"
CHROMA_COLLECTION_MITRE = "mitre_reference"

# ─── RAG Configuration ──────────────────────────────────────────────────────

RAG_TOP_K = 5               # Number of results to retrieve
RAG_MMR_LAMBDA = 0.7        # MMR diversity factor (0=diverse, 1=relevant)

# ─── PII Validation ─────────────────────────────────────────────────────────

PII_MAX_RETRIES = 3         # Max retries if PII is found in sanitized output

# ─── ZeroMQ (IPC) ───────────────────────────────────────────────────────────

ZMQ_SERVER_PORT = 5555
ZMQ_DASHBOARD_PORT = 5556

# ─── Company Node Assignments ───────────────────────────────────────────────

COMPANY_CONFIG = {
    "A": {
        "dataset": "nf-cse-cic-ids2018-v3",
        "partition": 0,
        "description": "Enterprise network — partition 1",
    },
    "B": {
        "dataset": "nf-cse-cic-ids2018-v3",
        "partition": 1,
        "description": "Enterprise network — partition 2",
    },
    "C": {
        "dataset": "gotham-2025",
        "partition": 0,
        "description": "IoT + C&C network traffic",
    },
}

# ─── Dataset Download URLs ──────────────────────────────────────────────────

DATASET_URLS = {
    "nf-cse-cic-ids2018-v3": {
        "source": "kaggle",
        "identifier": "dhoogla/nfcsecicids2018v3",
    },
    "gotham-2025": {
        "source": "kaggle",
        "identifier": "emilymuller/gotham-network-intrusion-detection-dataset",
    },
}
