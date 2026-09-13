from unittest import TestCase
from unittest.mock import patch

from scraper.proxy import (
    PlaywrightProxyRotator,
    choose_playwright_proxy,
    configured_proxy_urls,
    playwright_proxy_config,
)


class ProxyConfigurationTests(TestCase):
    @patch.dict(
        "os.environ",
        {
            "SCRAPER_PROXY_URLS": (
                "http://proxy-1.internal:8888,\n"
                "http://proxy-2.internal:8888"
            )
        },
        clear=False,
    )
    def test_configured_proxy_urls_accepts_commas_and_newlines(self):
        self.assertEqual(
            configured_proxy_urls(),
            (
                "http://proxy-1.internal:8888",
                "http://proxy-2.internal:8888",
            ),
        )

    @patch.dict(
        "os.environ",
        {
            "SCRAPER_PROXY_URLS": (
                "http://proxy-1.internal:8888,"
                "http://proxy-1.internal:8888"
            )
        },
        clear=False,
    )
    def test_configured_proxy_urls_removes_duplicates(self):
        self.assertEqual(
            configured_proxy_urls(),
            ("http://proxy-1.internal:8888",),
        )

    def test_playwright_proxy_config_separates_credentials(self):
        self.assertEqual(
            playwright_proxy_config(
                "http://proxy-user:proxy-pass@proxy.internal:8888"
            ),
            {
                "server": "http://proxy.internal:8888",
                "username": "proxy-user",
                "password": "proxy-pass",
            },
        )

    @patch.dict(
        "os.environ",
        {
            "SCRAPER_PROXY_URLS": (
                "http://proxy-1.internal:8888,"
                "http://proxy-2.internal:8888"
            )
        },
        clear=False,
    )
    def test_choose_playwright_proxy_selects_one_endpoint(
        self,
    ):
        self.assertIn(
            choose_playwright_proxy(),
            (
                {"server": "http://proxy-1.internal:8888"},
                {"server": "http://proxy-2.internal:8888"},
            ),
        )

    @patch("scraper.proxy.random.shuffle")
    def test_rotator_uses_all_proxies_before_repeating(self, _shuffle):
        rotator = PlaywrightProxyRotator(
            proxy_urls=(
                "http://proxy-1.internal:8888",
                "http://proxy-2.internal:8888",
                "http://proxy-3.internal:8888",
            )
        )

        selected = [
            rotator.choose()["server"]
            for _ in range(3)
        ]

        self.assertEqual(
            selected,
            [
                "http://proxy-1.internal:8888",
                "http://proxy-2.internal:8888",
                "http://proxy-3.internal:8888",
            ],
        )

    @patch("scraper.proxy.random.shuffle")
    def test_rotator_avoids_repeat_at_cycle_boundary(self, _shuffle):
        rotator = PlaywrightProxyRotator(
            proxy_urls=(
                "http://proxy-1.internal:8888",
                "http://proxy-2.internal:8888",
            )
        )

        selected = [
            rotator.choose()["server"]
            for _ in range(4)
        ]

        self.assertEqual(
            selected,
            [
                "http://proxy-1.internal:8888",
                "http://proxy-2.internal:8888",
                "http://proxy-1.internal:8888",
                "http://proxy-2.internal:8888",
            ],
        )

    @patch.dict(
        "os.environ",
        {
            "SCRAPER_PROXY_URLS": "",
        },
        clear=False,
    )
    def test_choose_playwright_proxy_allows_direct_mode(self):
        self.assertIsNone(
            choose_playwright_proxy()
        )

    def test_playwright_proxy_config_rejects_invalid_scheme(self):
        with self.assertRaises(ValueError):
            playwright_proxy_config(
                "ftp://proxy.internal:8888"
            )
