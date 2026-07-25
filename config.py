import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def require_env(name: str) -> str:
    value = os.getenv(name)

    if not value:
        raise RuntimeError(
            f"{name} is missing. Add it to {BASE_DIR / '.env'}"
        )

    return value


LOGIN_URL = require_env("LOGIN_URL")
BLS_EMAIL = require_env("BLS_EMAIL")
BLS_PASSWORD = require_env("BLS_PASSWORD")
