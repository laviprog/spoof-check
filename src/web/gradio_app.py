import gradio as gr
from typing import Tuple, Dict
import structlog

from src.services.audio import AudioService
from src.config import settings

log = structlog.get_logger()


class GradioApp:
    """
    Gradio web application for spoof detection.
    """

    def __init__(self):
        """Initialize the Gradio app."""
        self.audio_service = AudioService()
        log.info("GradioApp initialized")

    def process_audio(self, audio_file) -> Tuple[Dict, str, str]:
        """
        Process uploaded audio file and return results.
        """
        try:
            if audio_file is None:
                return {}, "Файл не загружен", ""

            log.info("Processing audio file", file=audio_file)

            # Detect spoof
            result = self.audio_service.detect_spoof(audio_file, cleanup=True)

            # Format probabilities for display
            probabilities = {
                "Подлинный": result["bonafide"],
                "Поддельный": result["spoof"],
            }

            # Format classification
            classification = result["classification"].upper()
            classification_color = "🟢" if classification == "BONAFIDE" else "🔴"
            classification_label = "Подлинный" if classification == "BONAFIDE" else "Поддельный"
            classification_text = f"{classification_color} **{classification_label}**"

            # Format detailed results
            details = ""
            for i, pred in enumerate(result["chunk_predictions"], 1):
                details += f"\n- **Фрагмент {i}**: Подлинный: {pred['bonafide']:.4f}, Поддельный: {pred['spoof']:.4f}"

            log.info(
                "Audio processed successfully",
                classification=classification,
                num_chunks=result["num_chunks"],
            )

            return probabilities, classification_text, details

        except Exception as e:
            log.error("Failed to process audio", error=str(e))
            error_msg = f"❌ **Ошибка**: {str(e)}"
            return {}, error_msg, ""

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

                Загрузите аудиофайл, чтобы определить, является ли он **подлинным** (настоящим) или **поддельным** (синтетическим/фейковым).

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

            # Connect the button to the processing function
            submit_btn.click(
                fn=self.process_audio,
                inputs=[audio_input],
                outputs=[probability_output, classification_output, details_output],
            )

            # Also trigger on audio upload
            audio_input.change(
                fn=self.process_audio,
                inputs=[audio_input],
                outputs=[probability_output, classification_output, details_output],
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
