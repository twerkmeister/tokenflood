import logging

WARN_ONCE_KEY = "warn_once_key"


class WarnOnceLogFilter(logging.Filter):
    def __init__(self):
        super().__init__()
        self._seen_warning_keys = set()

    def filter(self, record: logging.LogRecord) -> bool:
        if WARN_ONCE_KEY in record.__dict__:
            key = record.__dict__[WARN_ONCE_KEY]
            result = key not in self._seen_warning_keys
            self._seen_warning_keys.add(key)
            return result
        return True

    def clear(self):
        self._seen_warning_keys = set()


global_warn_once_filter = WarnOnceLogFilter()
