import os

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
    "roberta_data_path": os.path.join(PROJECT_ROOT, "core", "data", "processed", "roberta_data"),
    "model_save_path": os.path.join(PROJECT_ROOT, "core", "models", "roberta_absa"),
    "pseudo_model_path": os.path.join(PROJECT_ROOT, "core", "models", "roberta_absa_pseudo"),
    "unprocessed_aspects_unsup_path": os.path.join(PROJECT_ROOT, "core", "data", "processed", "unprocessed_aspects_unsup"),
    "processed_aspects_unsup_path": os.path.join(PROJECT_ROOT, "core", "data", "processed", "processed_aspects_unsup"),
    "pseudo_labeled_data": os.path.join(PROJECT_ROOT, "core", "data", "processed", "roberta_data", "pseudo_labeled_data.csv"),
    "synthetic_sentiments_path": os.path.join(PROJECT_ROOT, "core", "data", "processed", "synthetic_sentiments"),
    "aspect_clusters_path": os.path.join(PROJECT_ROOT, "core", "exploratory", "aspect_clusters_pca.csv"),
    "whitelist_path": os.path.join(PROJECT_ROOT, "core", "data", "whitelist.txt"),
    "useful_aspects_path": os.path.join(PROJECT_ROOT, "core", "data", "processed", "useful_aspects"),
    "roberta_final_path": os.path.join(PROJECT_ROOT, "core", "models", "roberta_absa_final"),
    "calibrated_model_path": os.path.join(PROJECT_ROOT, "core", "models", "roberta_absa_final", "calibrated"),
    "lcf_atepc_data_path": os.path.join(PROJECT_ROOT, "core", "data", "processed", "lcf_atepc_data"),
    "lcf_atepc_output_path": os.path.join(PROJECT_ROOT, "core", "models", "lcf_atepc")
}

# Ensure directories exist
for path in CONFIG.values():
    if path.endswith(os.sep) or not os.path.splitext(path)[1]:  # Check if it's a directory path
        os.makedirs(path, exist_ok=True)