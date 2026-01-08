"""
Worker module for processing raw images with RawForge.
Designed to be run in background threads by the FastAPI application.
"""
from RawForge.application.ModelHandler import ModelHandler
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ProcessConfig:
    """Configuration for a single image processing task."""
    in_file: str
    out_file: str
    tile_size: int = 256
    cfa: bool = False


class Worker:
    """Worker for processing raw images with a specific RawForge model."""

    def __init__(self, model: str, device: str | None = None):
        """
        Initialize a worker with a specific model.

        Args:
            model: Name of the RawForge model to use
            device: Optional device backend (cuda, cpu, mps)
        """
        self.model = model
        self.device = device
        self.handler = ModelHandler()

        # Load model during initialization
        logger.info(f"Loading model: {model}")
        self.handler.load_model(model)

        # Set device if specified
        if device:
            self.handler.set_device(device)

        logger.info(f"Worker initialized with model: {model}")

    def process(self, config: ProcessConfig) -> None:
        """
        Process a single raw image.

        Args:
            config: ProcessConfig with in_file, out_file, and options

        Raises:
            Exception: If any step of the processing fails
        """
        try:
            logger.info(f"Starting denoising: {config.in_file} -> {config.out_file}")

            # Load raw image and extract ISO
            iso = self.handler.load_rh(config.in_file)
            conditioning = [iso, 0]

            # Configure inference parameters
            inference_kwargs = {
                "disable_tqdm": True,  # Disable progress bar in background
                "tile_size": config.tile_size
            }

            # Run inference
            logger.info(f"Running inference on {config.in_file}")
            _, denoised_image = self.handler.run_inference(
                conditioning=conditioning,
                dims=None,
                inference_kwargs=inference_kwargs
            )

            # Save the result
            self.handler.handle_full_image(denoised_image, config.out_file, config.cfa)
            logger.info(f"Successfully saved denoised image to {config.out_file}")

        except Exception as e:
            logger.error(f"Failed to process {config.in_file}: {str(e)}", exc_info=True)
            raise
