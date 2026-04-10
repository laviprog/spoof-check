from src.logging import configure as configure_logging
from src.web.gradio_app import create_app

configure_logging()

import structlog
log = structlog.get_logger()


def main():
    log.info("Starting Audio Spoof Detection application")

    try:
        app = create_app()
        app.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
        )
    except KeyboardInterrupt:
        log.info("Application stopped by user")
    except Exception as e:
        log.error("Application failed", error=str(e))
        raise


if __name__ == "__main__":
    main()
