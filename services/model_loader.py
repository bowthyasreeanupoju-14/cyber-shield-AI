import pickle
from config import Config

# ── All models stored here after loading ──────────────────────────
_store = {}

def load_all_models():
    """Load all .pkl files once at app startup"""
    global _store
    print("Loading models...")

    with open(Config.FRAUD_MODEL_PATH,  'rb') as f: _store['model']    = pickle.load(f)
    with open(Config.FEATURES_PATH,     'rb') as f: _store['features'] = pickle.load(f)
    with open(Config.CONFIG_PATH,       'rb') as f: _store['config']   = pickle.load(f)
    with open(Config.SCALER_PATH,       'rb') as f: _store['scaler']   = pickle.load(f)
    with open(Config.ENCODERS_PATH,     'rb') as f: _store['encoders'] = pickle.load(f)
    with open(Config.METADATA_PATH,     'rb') as f: _store['metadata'] = pickle.load(f)
    with open(Config.SHAP_PATH,         'rb') as f: _store['shap']     = pickle.load(f)

    print(f"✅ Model loaded  : {_store['metadata']['model_type']}")
    print(f"✅ Accuracy      : {_store['metadata']['accuracy']*100:.2f}%")
    print(f"✅ AUC           : {_store['metadata']['auc']:.4f}")
    print(f"✅ Features      : {len(_store['features'])}")
    print(f"✅ SHAP explainer ready")

def get(key):
    """Get a loaded model/config by key"""
    return _store.get(key)
