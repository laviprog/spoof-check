import os
import uuid
from pathlib import Path
import structlog

from src.config import settings
from src.core.inference import SpoofDetector

log = structlog.get_logger()


class AudioService:
    """
    Service for handling audio file operations and spoof detection.
    """

    def __init__(self):
        """Initialize the audio service."""
        self.detector = SpoofDetector()
        self.files_dir = Path(settings.FILES_DIR)
        self.files_dir.mkdir(parents=True, exist_ok=True)
        log.info("AudioService initialized", files_dir=str(self.files_dir))

    def save_uploaded_file(self, file_path: str) -> str:
        """
        Save uploaded file to temporary directory.
        """
        try:
            # Generate unique filename
            file_extension = Path(file_path).suffix
            unique_filename = f"{uuid.uuid4()}{file_extension}"
            save_path = self.files_dir / unique_filename

            # Copy file
            import shutil
            shutil.copy2(file_path, save_path)

            log.info("File saved", original=file_path, saved=str(save_path))
            return str(save_path)

        except Exception as e:
            log.error("Failed to save file", error=str(e))
            raise

    def cleanup_file(self, file_path: str) -> None:
        """
        Remove temporary file.

        Args:
            file_path: Path to file to remove
        """
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                log.debug("File removed", path=file_path)
        except Exception as e:
            log.warning("Failed to remove file", path=file_path, error=str(e))

    def detect_spoof(self, audio_path: str, cleanup: bool = True) -> dict:
        """
        Detect spoof content in audio file.
        """
        saved_path = None
        try:
            # Save uploaded file
            saved_path = self.save_uploaded_file(audio_path)

            # Perform detection
            result = self.detector.predict(saved_path)

            # Add classification
            result["classification"] = (
                "spoof" if result["spoof"] > result["bonafide"] else "bonafide"
            )

            return result

        except Exception as e:
            log.error("Spoof detection failed", error=str(e))
            raise

        finally:
            # Cleanup if requested
            if cleanup and saved_path:
                self.cleanup_file(saved_path)
