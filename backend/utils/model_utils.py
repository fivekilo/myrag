import os
import logging
from pathlib import Path
import dotenv
from utils.paths import WORKSPACE_DIR

dotenv.load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Configure logger
logger = logging.getLogger(__name__)

def get_huggingface_model_path(model_name: str) -> str:
    """
    Convert a model name to a local path if the model exists locally.
    
    Args:
        model_name: The name of the model (e.g. "sentence-transformers/all-MiniLM-L6-v2")
        
    Returns:
        str: The local path to the model if it exists, otherwise returns the original model name
    """
    candidate_roots = []
    env_model_path = os.environ.get("HF_MODEL_PATH")
    if env_model_path:
        candidate_roots.append(Path(env_model_path))

    # Fall back to common local workspace-relative locations when the
    # backend process did not inherit the user's HF_MODEL_PATH env var.
    candidate_roots.extend(
        [
            WORKSPACE_DIR / "hfmodel",
            WORKSPACE_DIR.parent / "hfmodel",
        ]
    )

    for candidate_root in candidate_roots:
        if not candidate_root.exists():
            continue
        local_model_name = candidate_root.joinpath(*model_name.split("/"))
        if local_model_name.exists():
            logger.info(f"Using local model: {local_model_name}")
            return str(local_model_name)

    logger.info(f"Using remote model: {model_name}")
    return model_name
