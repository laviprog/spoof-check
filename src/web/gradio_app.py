from typing import Dict, Tuple

import gradio as gr
import structlog

from src.api.anti_spoofing import AntispoofingClient, AntiSpoofingResponse, PredictionResult
from src.config import settings
from src.services.audio import AudioService

log = structlog.get_logger()


class GradioApp:
    """
    Gradio web application for spoof detection.
    """

    def __init__(self):
        """Initialize the Gradio app."""
        self.audio_service = AudioService()
        self.antispoofing_client = AntispoofingClient()
        log.info("GradioApp initialized")

    async def process_audio(
        self,
        audio_file,
        model_source: str,
        hop_sec: float,
    ) -> Tuple[Dict[str, float], str, str]:
        """
        Process uploaded audio file and return results.
        """
        try:
            if audio_file is None:
                return {}, "Файл не загружен", ""

            log.info("Processing audio file", file=audio_file, model_source=model_source)

            if model_source == "Внешняя модель":
                return await self._process_with_external_model(audio_file, hop_sec)

            return self._process_with_local_model(audio_file)

        except Exception as e:
            log.error("Failed to process audio", error=str(e))
            error_msg = f"❌ **Ошибка**: {str(e)}"
            return {}, error_msg, ""

    def _process_with_local_model(self, audio_file) -> Tuple[Dict[str, float], str, str]:
        result = self.audio_service.detect_spoof(audio_file, cleanup=True)

        probabilities = {
            "Подлинный": result["bonafide"],
            "Поддельный": result["spoof"],
        }
        classification_text = self._format_classification(result["classification"])

        details = "**Источник модели**: локальная\n"
        for i, pred in enumerate(result["chunk_predictions"], 1):
            details += (
                f"\n- **Фрагмент {i}**: "
                f"Подлинный: {pred['bonafide']:.4f}, "
                f"Поддельный: {pred['spoof']:.4f}"
            )

        log.info(
            "Audio processed locally",
            classification=result["classification"],
            num_chunks=result["num_chunks"],
        )

        return probabilities, classification_text, details

    async def _process_with_external_model(
        self,
        audio_file,
        hop_sec: float,
    ) -> Tuple[Dict[str, float], str, str]:
        result = await self.antispoofing_client.predict(audio_file, hop_sec=hop_sec)
        prediction = result.global_prediction or self._aggregate_external_windows(result)

        probabilities = {
            "Подлинный": prediction.bonafide_probability,
            "Поддельный": prediction.spoof_probability,
        }
        classification_text = self._format_classification(prediction.label)
        details = self._format_external_details(result)

        log.info(
            "Audio processed by external API",
            classification=prediction.label,
            num_windows=len(result.windows),
            request_id=str(result.request_id),
        )

        return probabilities, classification_text, details

    @staticmethod
    def _aggregate_external_windows(result: AntiSpoofingResponse):
        if not result.windows:
            raise ValueError("External anti-spoofing service returned no window predictions")

        spoof_probability = sum(window.spoof_probability for window in result.windows) / len(
            result.windows
        )
        bonafide_probability = sum(window.bonafide_probability for window in result.windows) / len(
            result.windows
        )
        label = "spoof" if spoof_probability > bonafide_probability else "bonafide"
        confidence = max(spoof_probability, bonafide_probability)

        return PredictionResult(
            label=label,
            confidence=confidence,
            spoof_probability=spoof_probability,
            bonafide_probability=bonafide_probability,
        )

    @staticmethod
    def _format_classification(classification: str) -> str:
        classification = classification.upper()
        classification_color = "🟢" if classification == "BONAFIDE" else "🔴"
        classification_label = "Подлинный" if classification == "BONAFIDE" else "Поддельный"
        return f"{classification_color} **{classification_label}**"

    @staticmethod
    def _format_external_details(result: AntiSpoofingResponse) -> str:
        details = (
            "**Источник модели**: внешняя\n"
            f"**Request ID**: `{result.request_id}`\n"
            f"**Модель**: {result.processing.model_name} {result.processing.model_version}\n"
            f"**Время обработки**: {result.processing.processing_time_ms} мс\n"
        )
        for window in result.windows:
            details += (
                f"\n- **Фрагмент {window.window_index + 1}** "
                f"({window.start_sec:.2f}-{window.end_sec:.2f} сек): "
                f"Подлинный: {window.bonafide_probability:.4f}, "
                f"Поддельный: {window.spoof_probability:.4f}"
            )

        return details

    def create_interface(self) -> gr.Blocks:
        """
        Create and configure the Gradio interface.
        """
        with gr.Blocks(
            title="Определение поддельного аудио",
            theme=gr.themes.Soft(),
        ) as app:
            gr.Markdown(
                """
                # 🎵 Определение поддельного аудио

                Загрузите аудиофайл, чтобы определить, является ли он **подлинным**
                (настоящим) или **поддельным** (синтетическим/фейковым).

                **Поддерживаемые форматы**: WAV, MP3, FLAC, OGG.
                """
            )

            with gr.Row():
                with gr.Column(scale=1):
                    audio_input = gr.Audio(
                        label="Загрузите аудиофайл",
                        type="filepath",
                        sources=["upload"],
                    )

                    model_source = gr.Radio(
                        choices=["Локальная модель", "Внешняя модель"],
                        value="Локальная модель",
                        label="Источник модели",
                    )

                    hop_sec_slider = gr.Slider(
                        minimum=0.5,
                        maximum=10.0,
                        value=2.0,
                        step=0.5,
                        label="Шаг окна анализа (сек)",
                        info="Только для внешней модели",
                        visible=False,
                    )

                    model_source.change(
                        fn=lambda src: gr.update(visible=src == "Внешняя модель"),
                        inputs=model_source,
                        outputs=hop_sec_slider,
                    )

                    submit_btn = gr.Button(
                        "🔍 Проанализировать аудио",
                        variant="primary",
                        size="lg",
                    )

                    gr.Markdown(
                        """
                        ### ℹ️ Советы
                        - Загружайте чёткие аудиофайлы для лучших результатов
                        - Длинные файлы автоматически разбиваются на фрагменты
                        - Время обработки зависит от длительности файла
                        """
                    )

                with gr.Column(scale=1):
                    classification_output = gr.Markdown(
                        label="Результат обработки",
                        value="",
                    )

                    probability_output = gr.Label(
                        label="Вероятности",
                        num_top_classes=2,
                    )

                    details_output = gr.Markdown(
                        label="Детальный анализ",
                        value="",
                    )

            processing_inputs = [audio_input, model_source, hop_sec_slider]
            processing_outputs = [probability_output, classification_output, details_output]

            submit_btn.click(
                fn=self.process_audio,
                inputs=processing_inputs,
                outputs=processing_outputs,
            )

            audio_input.change(
                fn=self.process_audio,
                inputs=processing_inputs,
                outputs=processing_outputs,
            )

            model_source.change(
                fn=self.process_audio,
                inputs=processing_inputs,
                outputs=processing_outputs,
            )

        return app

    def launch(
        self,
        server_name: str = "0.0.0.0",
        server_port: int = 7860,
        share: bool = False,
        **kwargs,
    ):
        """
        Launch the Gradio application.
        """
        app = self.create_interface()

        log.info(
            "Launching Gradio app",
            server_name=server_name,
            server_port=server_port,
            root_path=settings.ROOT_PATH,
        )

        app.launch(
            server_name=server_name,
            server_port=server_port,
            share=share,
            root_path=settings.ROOT_PATH if settings.ROOT_PATH != "/" else None,
            **kwargs,
        )


def create_app() -> GradioApp:
    """
    Factory function to create a GradioApp instance.
    """
    return GradioApp()
