import logging
from typing import Dict, Any, Optional, Union

import torch
from allennlp.training.callbacks.log_writer import LogWriterCallback

logger = logging.getLogger(__name__)


class MetricsLoggerCallback(LogWriterCallback):
    """
    Callback for logging metrics while training model.
    """
    def _to_params(self) -> Dict[str, Any]:
        pass

    def log_tensors(self, tensors: Dict[str, torch.Tensor], log_prefix: str = "", epoch: Optional[int] = None) -> None:
        pass

    def log_scalars(
        self,
        scalars: Dict[str, Union[int, float]],
        log_prefix: str = "",
        epoch: Optional[int] = None,
    ) -> None:
        """
        Metrics logger
        """
        metrics = scalars
        metrics['current_epoch'] = self.trainer._epochs_completed + 1
        metrics['batches_completed'] = self.trainer._batches_in_epoch_completed
        metrics['total_batches_completed'] = self.trainer._total_batches_completed

        logger.info(','.join(f'{k}={v}' for k, v in metrics.items()))
