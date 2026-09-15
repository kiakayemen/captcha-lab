from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, TypeVar

from django.core.exceptions import SynchronousOnlyOperation
from django.db import close_old_connections


Result = TypeVar("Result")

_writer_state = threading.local()
_writer = ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="scraper-database-writer",
)


def _run_on_database_thread(
    operation: Callable[[], Result],
) -> Result:
    close_old_connections()
    _writer_state.active = True
    try:
        return operation()
    finally:
        _writer_state.active = False
        close_old_connections()


def run_database_write(
    operation: Callable[[], Result],
) -> Result:
    """Run an ORM write safely when Playwright owns an async context.

    Normal Django execution stays on the caller thread.  Playwright's
    synchronous API can make Django detect a running async loop; in that
    case retry the untouched operation on a dedicated database thread.
    """
    if getattr(_writer_state, "active", False):
        return operation()

    try:
        return operation()
    except SynchronousOnlyOperation:
        return _writer.submit(
            _run_on_database_thread,
            operation,
        ).result()
