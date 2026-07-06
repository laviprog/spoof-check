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

    DEVICE: str = "cpu"  # Device to run inference on (e.g., "cpu", "cuda", "cuda:0")
    COMPUTE_TYPE: str = "float32"
    MODELS_DIR: str = "models"  # Directory for storing models

    ANTISPOOFING_BASE_URL: str | None = None
    ANTISPOOFING_USERNAME: str | None = None
    ANTISPOOFING_PASSWORD: str | None = None

    @property
    def antispoofing_enabled(self) -> bool:
        """External anti-spoofing model is available when its base URL is configured."""
        return bool(self.ANTISPOOFING_BASE_URL)

    @property
    def antispoofing_auth_enabled(self) -> bool:
        """Authentication is used only when both username and password are configured."""
        return bool(self.ANTISPOOFING_USERNAME and self.ANTISPOOFING_PASSWORD)


settings = Settings()
