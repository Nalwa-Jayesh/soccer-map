import logging
from typing import Any, Dict

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class GPUManager:
    def __init__(self):
        self.device = None
        self.device_name = None
        self.gpu_available = False
        self.backend_type = None
        self._detect_gpu()

    def _detect_gpu(self):
        try:
            import torch

            if torch.cuda.is_available():
                self.device = "cuda"
                self.device_name = torch.cuda.get_device_name(0)
                self.gpu_available = True
                self.backend_type = "CUDA"
                logger.info(f"✅ CUDA GPU detected: {self.device_name}")
                logger.info(f"   CUDA Version: {torch.version.cuda}")
                logger.info(
                    f"   GPU Memory: {
                        torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
                )

            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
                self.device_name = "Apple Silicon GPU"
                self.gpu_available = True
                self.backend_type = "MPS"
                logger.info("✅ Apple Silicon GPU detected (MPS)")

            else:
                self.device = "cpu"
                self.device_name = "CPU"
                self.gpu_available = False
                self.backend_type = "CPU"
                logger.info("⚠️  No GPU detected, using CPU")

        except ImportError:
            logger.warning("PyTorch not available, using CPU")
            self.device = "cpu"
            self.device_name = "CPU"
            self.gpu_available = False
            self.backend_type = "CPU"

    def get_device_info(self) -> Dict[str, Any]:
        info = {
            "device": self.device,
            "device_name": self.device_name,
            "gpu_available": self.gpu_available,
            "backend_type": self.backend_type,
        }

        if self.backend_type == "CUDA":
            try:
                import torch

                info.update(
                    {
                        "cuda_version": torch.version.cuda,
                        "gpu_count": torch.cuda.device_count(),
                        "current_gpu": torch.cuda.current_device(),
                        "gpu_memory_total": torch.cuda.get_device_properties(
                            0
                        ).total_memory,
                        "gpu_memory_allocated": torch.cuda.memory_allocated(0),
                        "gpu_memory_cached": torch.cuda.memory_reserved(0),
                    }
                )
            except:
                pass

        return info

    def optimize_gpu_settings(self):
        if self.backend_type == "CUDA":
            try:
                import torch

                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                logger.info("CUDA optimizations enabled")
            except:
                pass
