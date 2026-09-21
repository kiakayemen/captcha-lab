import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR / ".env.secrets")


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


@dataclass(frozen=True)
class BLSAccount:
    email: str
    password: str


def configured_bls_accounts() -> tuple[BLSAccount, ...]:
    """Load complete account pairs, preserving the original account settings."""
    accounts = [BLSAccount(BLS_EMAIL, BLS_PASSWORD)]
    email_2 = os.getenv("BLS_EMAIL_2", "").strip()
    password_2 = os.getenv("BLS_PASSWORD_2", "")
    if bool(email_2) != bool(password_2):
        raise RuntimeError("BLS_EMAIL_2 and BLS_PASSWORD_2 must both be set")
    if email_2:
        if email_2 == BLS_EMAIL:
            raise RuntimeError("BLS_EMAIL_2 must differ from BLS_EMAIL")
        accounts.append(BLSAccount(email_2, password_2))
    return tuple(accounts)
