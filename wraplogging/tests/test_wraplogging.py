import logging

import wraplogging.wraplogging


def test_app_logger(caplog):
    """Test that app_logger logs correctly."""
    with caplog.at_level(logging.DEBUG):
        app_logger = wraplogging.wraplogging.create_logger("app_logger", level=logging.DEBUG, show_time=True)
        app_logger.debug("Test debug message")
        app_logger.info("Test info message")

    # ✅ Check if logs exist in captured output
    assert "Test debug message" in caplog.text
    assert "Test info message" in caplog.text


def test_db_logger(caplog):
    """Test that db_logger logs correctly."""
    with caplog.at_level(logging.WARNING):
        db_logger = wraplogging.wraplogging.create_logger(
            "db_logger", level=logging.WARNING, show_time=False, show_level=False
        )
        db_logger.warning("Database warning test")
        db_logger.error("Database error test")

    # ✅ Check if logs exist in captured output
    assert "Database warning test" in caplog.text
    assert "Database error test" in caplog.text


def test_logger_reconfiguration(caplog):
    """Test that reconfiguring a logger changes its behavior."""
    db_logger = wraplogging.wraplogging.create_logger("db_logger", level=logging.WARNING)
    db_logger.warning("Old warning message")

    # ✅ Reconfigure db_logger to only log ERROR and above
    db_logger = wraplogging.wraplogging.create_logger("db_logger", level=logging.ERROR)
    db_logger.warning("This should not appear")
    db_logger.error("This error should appear")

    # ✅ Check that old warning appears, but the second one doesn't
    assert "Old warning message" in caplog.text
    assert "This should not appear" not in caplog.text
    assert "This error should appear" in caplog.text
