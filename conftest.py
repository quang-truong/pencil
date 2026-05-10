import pytest


def pytest_collection_modifyitems(config, items):
    # Always exclude distributed tests from collection
    items[:] = [
        item
        for item in items
        if getattr(item.fspath, "basename", "") != "test_distributed.py"
    ]

    # If user explicitly selected the data marker, do not skip
    selected_markexpr = str(
        getattr(config, "option", None) and getattr(config.option, "markexpr", "")
    )
    if selected_markexpr and "data" in selected_markexpr:
        return

    skip_data = pytest.mark.skip(
        reason="skipped: requires dataset downloads/cache; run with -m data to include"
    )
    for item in items:
        if item.get_closest_marker("data") is not None:
            item.add_marker(skip_data)
