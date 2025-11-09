import logging

from tokenflood.logging import WARN_ONCE_KEY, WarnOnceLogFilter


def test_warnings_filter_log_once(caplog):
    logger = logging.getLogger("test_warnings_filter_log_once")
    logger.addFilter(WarnOnceLogFilter())
    warning_key_1 = "bad_output_tokens"
    warning_msg_1 = "Too many output tokens."

    with caplog.at_level(logging.WARNING):
        logger.warning(warning_msg_1, extra={WARN_ONCE_KEY: warning_key_1})
    assert warning_msg_1 in caplog.text

    caplog.clear()

    with caplog.at_level(logging.WARNING):
        logger.warning(warning_msg_1, extra={WARN_ONCE_KEY: warning_key_1})
    assert caplog.text == ""


def test_warnings_filter_log_reset(caplog):
    logger = logging.getLogger("test_warnings_filter_log_reset")
    log_filter = WarnOnceLogFilter()
    logger.addFilter(log_filter)
    warning_key_1 = "bad_output_tokens"
    warning_msg_1 = "Too many output tokens."

    with caplog.at_level(logging.WARNING):
        logger.warning(warning_msg_1, extra={WARN_ONCE_KEY: warning_key_1})
    assert warning_msg_1 in caplog.text

    caplog.clear()
    log_filter.clear()

    with caplog.at_level(logging.WARNING):
        logger.warning(warning_msg_1, extra={WARN_ONCE_KEY: warning_key_1})
    assert warning_msg_1 in caplog.text


def test_warnings_filter_different_keys(caplog):
    logger = logging.getLogger("test_warnings_filter_different_keys")
    logger.addFilter(WarnOnceLogFilter)
    warning_key_1 = "bad_output_tokens"
    warning_msg_1 = "Too many output tokens."
    warning_key_2 = "bad_output_tokens_specific"

    with caplog.at_level(logging.WARNING):
        logging.warning(warning_msg_1, extra={WARN_ONCE_KEY: warning_key_1})
    assert warning_msg_1 in caplog.text

    caplog.clear()

    with caplog.at_level(logging.WARNING):
        logging.warning(warning_msg_1, extra={WARN_ONCE_KEY: warning_key_2})
    assert warning_msg_1 in caplog.text
