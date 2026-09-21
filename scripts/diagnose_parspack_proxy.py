"""Report the proxy's HTTPS CONNECT response without exposing credentials."""

import base64
import os
import socket
from urllib.parse import unquote, urlsplit


raw_url = os.environ.get("PARSPACK_PROXY_URL")
if not raw_url:
    raise SystemExit("PARSPACK_PROXY_URL is not set on this worker")

proxy = urlsplit(raw_url)
if proxy.scheme != "http" or not proxy.hostname or not proxy.port:
    raise SystemExit("PARSPACK_PROXY_URL must be an HTTP proxy URL with a port")

auth_header = b""
if proxy.username is not None:
    credentials = "{}:{}".format(
        unquote(proxy.username), unquote(proxy.password or "")
    )
    token = base64.b64encode(credentials.encode("utf-8"))
    auth_header = b"Proxy-Authorization: Basic " + token + b"\r\n"

for target in ("api.ipify.org", "iran.blsspainglobal.com"):
    try:
        with socket.create_connection((proxy.hostname, proxy.port), timeout=15) as sock:
            sock.settimeout(15)
            request = (
                f"CONNECT {target}:443 HTTP/1.1\r\n"
                f"Host: {target}:443\r\n"
            ).encode("ascii") + auth_header + b"\r\n"
            sock.sendall(request)
            response = bytearray()
            while b"\r\n\r\n" not in response and len(response) < 16384:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                response.extend(chunk)
        lines = response.split(b"\r\n\r\n", 1)[0].decode(
            "iso-8859-1", errors="replace"
        ).split("\r\n")
        status = lines[0] if lines else "no HTTP response"
        fields = {}
        for line in lines[1:]:
            name, separator, value = line.partition(":")
            if separator and name.lower() in {"server", "proxy-authenticate", "via"}:
                fields[name.lower()] = value.strip()
        print(target, status, fields)
    except (OSError, ValueError) as error:
        print(target, type(error).__name__, str(error))
