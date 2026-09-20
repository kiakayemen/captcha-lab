"""Test one proxy's HTTPS tunneling before adding it to scraper rotation."""

import getpass
from urllib.parse import quote

from django.core.management.base import BaseCommand, CommandError

from scraper.proxy import probe_proxy_tunnel


class Command(BaseCommand):
    help = "Check one authenticated proxy without running the scraper."

    def add_arguments(self, parser):
        parser.add_argument("--server", required=True, help="Proxy host:port")
        parser.add_argument("--username", default="captcha")

    def handle(self, *args, **options):
        server = options["server"].strip()
        if not server or any(character in server for character in "/@?#"):
            raise CommandError("--server must be a plain host:port, without credentials")
        password = getpass.getpass("Proxy password: ")
        if not password:
            raise CommandError("Proxy password cannot be empty")
        proxy_url = (
            f"http://{quote(options['username'], safe='')}:"
            f"{quote(password, safe='')}@{server}"
        )
        for target in ("api.ipify.org", "iran.blsspainglobal.com"):
            try:
                probe_proxy_tunnel(proxy_url, target)
            except Exception as error:
                raise CommandError(
                    f"HTTPS CONNECT to {target}:443 through {server} failed: "
                    f"{type(error).__name__}: {error}"
                ) from None
            self.stdout.write(self.style.SUCCESS(
                f"HTTPS CONNECT to {target}:443 through {server} succeeded"
            ))
