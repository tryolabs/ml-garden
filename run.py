import argparse
import logging

from explainerdashboard import ExplainerDashboard

from pipeline_lib import Pipeline

# Constants
DEFAULT_LOG_LEVEL = "INFO"
LOG_LEVEL_CHOICES = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
DEFAULT_PORT = 8050


def setup_logging(log_level: int) -> None:
    """
    Set up logging configuration.

    Parameters
    ----------
    log_level : int
        Logging level as defined in the logging module.
    """
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    """
    Main function to run the pipeline based on command line arguments.
    """
    parser = argparse.ArgumentParser(description="Run the pipeline.")
    parser.add_argument("config_file", type=str, help="Path to the configuration JSON file.")
    parser.add_argument(
        "--predict", action="store_true", help="Run the pipeline in prediction mode."
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=DEFAULT_LOG_LEVEL,
        choices=LOG_LEVEL_CHOICES,
        help=f"Set the log level (default: {DEFAULT_LOG_LEVEL})",
    )
    parser.add_argument("--explainer", action="store_true", help="Run the explainer dashboard.")
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Port for the explainer dashboard (default: {DEFAULT_PORT}).",
    )
    args = parser.parse_args()

    # Set up logging
    log_level = getattr(logging, args.log_level.upper())
    setup_logging(log_level)

    logging.info(f"Loading pipeline from {args.config_file}")
    pipeline = Pipeline.from_json(args.config_file)

    if args.predict:
        pipeline.predict()
    else:
        data = pipeline.train()
        if args.explainer and data.explainer is not None:
            ExplainerDashboard(explainer=data.explainer).run(port=args.port)


if __name__ == "__main__":
    main()
