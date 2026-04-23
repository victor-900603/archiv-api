import logging
import os

DEFAULT_LOG_FORMAT = "%(asctime)s[%(levelname)s][%(name)s]%(message)s"
DEFAULT_LOG_LEVEL = "DEBUG"


def configure_logging(log_level: str | None = None) -> None:
    level_name = (log_level or os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL)).upper()
    level = getattr(logging, level_name, logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    if root_logger.handlers:
        for handler in root_logger.handlers:
            handler.setLevel(level)
        return

    logging.basicConfig(
        level=level,
        format=DEFAULT_LOG_FORMAT,
    )
    
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("groq").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)