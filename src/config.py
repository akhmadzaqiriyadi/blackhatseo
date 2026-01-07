import os
import json
import logging
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
CONFIG_DIR = BASE_DIR / "config"

# Ensure directories exist
for dir_path in [DATA_DIR, MODELS_DIR, LOGS_DIR, CONFIG_DIR]:
    dir_path.mkdir(exist_ok=True)

# Default Config Path
DEFAULT_CONFIG_PATH = CONFIG_DIR / "config.json"

def load_config(config_path=None):
    """Load configuration from JSON file."""
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    
    config_path = Path(config_path)
    if not config_path.exists():
        return {'use_selenium': False, 'use_bert': False, 'verbose': False}
    
    with open(config_path, 'r') as f:
        return json.load(f)

def setup_logging(log_file="app.log", verbose=False):
    """Configure logging."""
    log_path = LOGS_DIR / log_file
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(str(log_path))
        ]
    )
    return logging.getLogger("BlackHatSEO")
