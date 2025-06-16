import os
from dotenv import load_dotenv
load_dotenv()

def get_project_root():
    """
    Get the absolute path to the project root directory.
    
    Returns:
        str: Absolute path to the project root directory
    """
    # Config file is in 'core/config', so root is up two directories
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Get project root once to use in all paths
PROJECT_ROOT = get_project_root()

CONFIG = {
    "base_path": os.path.join(PROJECT_ROOT, "core", "data", "raw"),
    "processed_path": os.path.join(PROJECT_ROOT, "core", "data", "processed"),
    "supervised_path": os.path.join(PROJECT_ROOT, "core", "data", "processed", "supervised_data.csv"),
    "unsupervised_path": os.path.join(PROJECT_ROOT, "core", "data", "processed", "unsupervised_data.csv"),
    "unprocessed_aspects_path": os.path.join(PROJECT_ROOT, "core", "data", "processed", "unprocessed_aspects"),
    "processed_aspects_path": os.path.join(PROJECT_ROOT, "core", "data", "processed", "processed_aspects"),
    "roberta_save_path": os.path.join(PROJECT_ROOT, "core", "models", "roberta_absa"),
    "pseudo_roberta_model_path": os.path.join(PROJECT_ROOT, "core", "models", "roberta_absa_pseudo"),
    "synthetic_sentiments_path": os.path.join(PROJECT_ROOT, "core", "data", "processed", "synthetic_sentiments"),
    "whitelist_path": os.path.join(PROJECT_ROOT, "core", "data", "whitelist.txt"),
    "useful_aspects_path": os.path.join(PROJECT_ROOT, "core", "data", "processed", "useful_aspects"),
    "roberta_final_path": os.path.join(PROJECT_ROOT, "core", "models", "roberta_absa_final"),
    "lcf_atepc_data_path": os.path.join(PROJECT_ROOT, "core", "data", "processed", "lcf_atepc_data"),
    "lcf_atepc_output_path": os.path.join(PROJECT_ROOT, "core", "models", "lcf_atepc"),
    "telegram_bot_token": os.getenv("TELEGRAM_BOT_TOKEN")
}

# Ensure directories exist
for key, path in CONFIG.items():
    # Skip tokens and any non-path values
    if "path" in key and (path.endswith(os.sep) or not os.path.splitext(path)[1]):
        os.makedirs(path, exist_ok=True)