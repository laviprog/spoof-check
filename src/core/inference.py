import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict
import structlog

from src.core.model import Spectra0Model
from src.config import settings

log = structlog.get_logger()


class SpoofDetector:
    """
    Detector for audio spoof content using the Spectra0 model.
    """

    def __init__(
        self,
        model_name: str = "MTUCI/spectra_0",
        device: str = None,
        threshold: float = -1.0625009,
        chunk_duration: float = 4.0,
        overlap: float = 0.5,
    ):
        """
        Initialize the spoof detector.
        """
        self.model_name = model_name
        self.threshold = threshold
        self.chunk_duration = chunk_duration
        self.overlap = overlap
        self.sample_rate = 16000  # Required sample rate for the model

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        log.info(
            "Initializing SpoofDetector",
            model=model_name,
            device=str(self.device),
            threshold=threshold,
        )

        # Load model
        self.model = self._load_model()
        self.model.eval()

        log.info("SpoofDetector initialized successfully")

    def _load_model(self) -> Spectra0Model:
        """Load the Spectra0 model from HuggingFace or local directory."""
        try:
            # Try loading from local directory first
            local_model_path = Path(settings.MODELS_DIR) / "spectra_0"
            if local_model_path.exists():
                log.info("Loading model from local directory", path=str(local_model_path))
                model = Spectra0Model.from_pretrained(str(local_model_path))
            else:
                log.info("Downloading model from HuggingFace", model=self.model_name)
                model = Spectra0Model.from_pretrained(self.model_name)
                # Save to local directory for future use
                local_model_path.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(str(local_model_path))
                log.info("Model saved to local directory", path=str(local_model_path))

            return model.to(self.device)
        except Exception as e:
            log.error("Failed to load model", error=str(e))
            raise

    def _load_audio(self, audio_path: str) -> torch.Tensor:
        """
        Load and preprocess audio file.
        """
        try:
            # Load audio
            waveform, sr = torchaudio.load(audio_path)

            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Resample if needed
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)

            # Remove channel dimension
            waveform = waveform.squeeze(0)

            log.debug(
                "Audio loaded",
                duration=waveform.shape[0] / self.sample_rate,
                sample_rate=self.sample_rate,
            )

            return waveform
        except Exception as e:
            log.error("Failed to load audio", path=audio_path, error=str(e))
            raise

    def _split_audio_into_chunks(self, waveform: torch.Tensor) -> List[torch.Tensor]:
        """
        Split audio into overlapping chunks.
        """
        chunk_samples = int(self.chunk_duration * self.sample_rate)
        step_samples = int(chunk_samples * (1 - self.overlap))

        chunks = []
        total_samples = waveform.shape[0]

        # If audio is shorter than chunk duration, pad it
        if total_samples < chunk_samples:
            padding = chunk_samples - total_samples
            waveform = torch.nn.functional.pad(waveform, (0, padding))
            chunks.append(waveform)
            log.debug("Audio padded", original_samples=total_samples, padded_samples=chunk_samples)
        else:
            # Split into overlapping chunks
            start = 0
            while start + chunk_samples <= total_samples:
                chunk = waveform[start : start + chunk_samples]
                chunks.append(chunk)
                start += step_samples

            # Handle remaining samples if any
            if start < total_samples:
                remaining = waveform[start:]
                padding = chunk_samples - remaining.shape[0]
                remaining = torch.nn.functional.pad(remaining, (0, padding))
                chunks.append(remaining)

        log.debug("Audio split into chunks", num_chunks=len(chunks))
        return chunks

    @torch.inference_mode()
    def _predict_chunk(self, chunk: torch.Tensor) -> Tuple[float, float]:
        """
        Predict spoof probability for a single chunk.
        """
        # Add batch dimension
        chunk = chunk.unsqueeze(0).to(self.device)

        # Get logits
        logits = self.model(chunk)

        # Apply softmax to get probabilities
        probs = torch.nn.functional.softmax(logits, dim=1)

        spoof_prob = probs[0, 0].item()
        bonafide_prob = probs[0, 1].item()

        return bonafide_prob, spoof_prob

    def predict(self, audio_path: str) -> Dict[str, float]:
        """
        Predict spoof probabilities for an audio file.
        """
        log.info("Starting prediction", audio_path=audio_path)

        try:
            # Load audio
            waveform = self._load_audio(audio_path)

            # Split into chunks
            chunks = self._split_audio_into_chunks(waveform)

            # Predict for each chunk
            bonafide_probs = []
            spoof_probs = []

            for i, chunk in enumerate(chunks):
                bonafide_prob, spoof_prob = self._predict_chunk(chunk)
                bonafide_probs.append(bonafide_prob)
                spoof_probs.append(spoof_prob)
                log.debug(
                    "Chunk prediction",
                    chunk_idx=i,
                    bonafide=bonafide_prob,
                    spoof=spoof_prob,
                )

            # Average predictions
            avg_bonafide = np.mean(bonafide_probs)
            avg_spoof = np.mean(spoof_probs)

            result = {
                "bonafide": float(avg_bonafide),
                "spoof": float(avg_spoof),
                "num_chunks": len(chunks),
                "chunk_predictions": [
                    {"bonafide": b, "spoof": s}
                    for b, s in zip(bonafide_probs, spoof_probs)
                ],
            }

            log.info(
                "Prediction completed",
                bonafide=avg_bonafide,
                spoof=avg_spoof,
                num_chunks=len(chunks),
            )

            return result

        except Exception as e:
            log.error("Prediction failed", error=str(e))
            raise

    def classify(self, audio_path: str) -> str:
        """
        Classify audio as bonafide or spoof.

        """
        result = self.predict(audio_path)
        classification = "spoof" if result["spoof"] > result["bonafide"] else "bonafide"

        log.info("Classification result", classification=classification)
        return classification
