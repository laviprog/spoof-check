from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables or a .env file.
    """

    model_config = SettingsConfigDict(env_file=".env")

    LOG_LEVEL: str = "DEBUG"  # DEBUG | INFO | WARNING | ERROR | CRITICAL
    ENV: str = "dev"  # dev | prod
    FILES_DIR: str = "data/files"  # Directory for uploaded files
    ROOT_PATH: str = "/spoof-check"  # WEB root path

    DEVICE: str = "cpu" # Device to run inference on (e.g., "cpu", "cuda", "cuda:0")
    COMPUTE_TYPE: str = "float32"
    MODELS_DIR: str = "models"  # Directory for storing models




settings = Settings()
