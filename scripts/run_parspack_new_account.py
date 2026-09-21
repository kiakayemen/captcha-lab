"""Run the scraper on the deployed worker using account 2 and ParsPack only.

Run from the project root or any directory:
    python scripts/run_parspack_new_account.py

The worker must already have BLS_EMAIL_2, BLS_PASSWORD_2 and either a
credential-bearing ParsPack proxy URL or TINYPROXY_USERNAME/PASSWORD.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARSPACK_HOST = "94.184.43.30"


def parspack_proxy_url(env: dict[str, str]) -> str:
    configured = env.get("PARSPACK_PROXY_URL", "").strip()
    candidates = [configured] if configured else [
        item.strip()
        for item in env.get("SCRAPER_PROXY_URLS", "").replace("\n", ",").split(",")
        if item.strip()
    ]
    matches = [url for url in candidates if urlsplit(url).hostname == PARSPACK_HOST]
    if not matches and not candidates:
        matches = [f"http://{PARSPACK_HOST}:8888"]
    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one ParsPack proxy URL for 94.184.43.30 in "
            "PARSPACK_PROXY_URL or SCRAPER_PROXY_URLS"
        )
    parsed = urlsplit(matches[0])
    if parsed.scheme != "http" or not parsed.port:
        raise ValueError("ParsPack proxy URL must be HTTP and include a port")
    if parsed.username and parsed.password:
        return matches[0]
    username = env.get("TINYPROXY_USERNAME", "")
    password = env.get("TINYPROXY_PASSWORD", "")
    if not username or not password:
        raise ValueError("Tinyproxy username/password must be in the proxy URL or worker environment")
    if parsed.username or parsed.password:
        raise ValueError("ParsPack proxy URL has incomplete Tinyproxy credentials")
    authority = f"{quote(username, safe='')}:{quote(password, safe='')}@{PARSPACK_HOST}:{parsed.port}"
    return urlunsplit(("http", authority, "", "", ""))


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    load_dotenv(PROJECT_ROOT / ".env.secrets")
    env = os.environ.copy()
    email = env.get("BLS_EMAIL_2", "").strip()
    password = env.get("BLS_PASSWORD_2", "")
    if not email or not password:
        print("BLS_EMAIL_2 and BLS_PASSWORD_2 must both be set on the worker.", file=sys.stderr)
        return 2
    try:
        proxy_url = parspack_proxy_url(env)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2

    # The child process sees only the requested account and proxy. Never print
    # the URL: it contains the Tinyproxy password.
    env["BLS_EMAIL"] = email
    env["BLS_PASSWORD"] = password
    env.pop("BLS_EMAIL_2", None)
    env.pop("BLS_PASSWORD_2", None)
    env["SCRAPER_PROXY_URLS"] = proxy_url
    print("Starting scraper with account 2 through the ParsPack proxy.", flush=True)
    return subprocess.call(
        [sys.executable, "manage.py", "run_scraper", "--headless", "--allow-single-proxy"],
        cwd=PROJECT_ROOT,
        env=env,
    )


if __name__ == "__main__":
    raise SystemExit(main())
