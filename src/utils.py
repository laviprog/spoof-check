from src.config import settings


def is_dev_env() -> bool:
    """
    Check if the current environment is development.
    """
    return settings.ENV == "dev"
