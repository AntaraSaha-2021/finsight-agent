"""
Regression Risk: Any change to .env or .settings.py could break all downstream components silently
"""

import pytest
import os

def test_settings_load():
    from config import settings
    assert settings is not None

def test_required_values_present():
    from config import settings
    assert settings.GROQ_API_KEY, "GROQ_API_KEY is missing from .env"
    assert settings.GROQ_MODEL, "GROQ_MODEL is missing from .env"
    assert settings.EMBED_MODEL, "EMBED_MODEL is missing from .env"

def test_chunk_size_is_integer():
    from config import settings
    assert isinstance(settings.CHUNK_SIZE, int)
    assert settings.CHUNK_SIZE>0

def test_data_dir_exists():
    from config import settings
    assert os.path.exists(settings.DATA_DIR), \
    f"data/ directory missing at {settings.DATA_DIR}"

def test_api_secret_key_is_strong():
    from config import settings
    assert len(settings.API_SECRET_KEY)>=32, \
    "API_SECRET_KEY is too short - run: python -c \"import secrets; print(secrets.token_hex(32))\""