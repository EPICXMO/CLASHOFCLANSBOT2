from actions.adb import ADBController


def test_adb_dry_run_basics():
    c = ADBController(dry_run=True)
    assert c.connect() is False
    w, h = c.window_size()
    assert w > 0 and h > 0
    c.click(100, 100)  # no exception
    c.swipe(10, 10, 20, 20)
    c.tap_norm(0.5, 0.5)
    c.swipe_norm(0.1, 0.1, 0.2, 0.2)

