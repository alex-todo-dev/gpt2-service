"""
Configuration settings
"""

import os

# Model settings
MODEL_NAME: str = os.getenv('MODEL_NAME', 'gpt2')
PAD_TOKEN_ID: int = 50256

# Generation defaults
DEFAULT_MAX_NEW_TOKENS: int = 50
MAX_MAX_NEW_TOKENS: int = 1024
MIN_MAX_NEW_TOKENS: int = 1

DEFAULT_NUM_RETURN_SEQUENCES: int = 5
MIN_NUM_RETURN_SEQUENCES: int = 0
MAX_NUM_RETURN_SEQUENCES: int = 10

DEFAULT_TEMP: float = 0.7
MIN_TEMP: float = 0.0
MAX_TEMP: float = 1.0

DEFAULT_TOP_P: float = 0.9
MIN_TOP_P: float = 0.0
MAX_TOP_P: float = 1.0

# Text validation settings
MIN_TEXT_LENGTH: int = 1

# Token validation settings
MIN_TOKENS_LENGTH: int = 1
MAX_TOKENS_LENGTH: int = 1024

