import torch
from pytorch_lightning import LightningModule
from collections import OrderedDict
import torch.optim as optim
from typing import Any, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT


class ModelLearner(LightningModule):
    def __init__(
            self, model, scenario, train_config, scenario_config, experiment_config, model_name=None
    ):
        super(ModelLearner, self).__init__()

        self.model = model
        self.scenario = scenario
        self.experiment_config = experiment_config
        self.train_config = train_config
        self.scenario_config = scenario_config
        self.model_name = model_name

    # ---
    # --- Pytorch ligthning --- #
    def configure_optimizers(self):
        optimiser = optim.Adam(
            self.model.parameters(), lr=self.train_config["learning_rate"]
        )
        return optimiser

    def training_step(self, batch, batch_idx):
        # Forward pass through the encoders
        # if self.model_name in ("gmc4", "gmc5", "gmc6", "gmc7", "gmc8", "gmc9", "gmc10"):
        if "gmc" in self.model_name and "gmc" != self.model_name:
            data = batch[0], batch[1]
        else: data = batch[0]
        loss, tqdm_dict = self.model.training_step(data, self.train_config)
        output = OrderedDict(
            {"loss": loss, "progress_bar": tqdm_dict, "log": tqdm_dict}
        )
        self.log_dict(tqdm_dict)
        return output

    def training_epoch_end(self, outputs):
        log_keys = list(outputs[0]["log"].keys())
        for log_key in log_keys:
            avg_batch_log = (
                torch.stack(
                    [
                        outputs[batch_output_idx]["log"][log_key]
                        for batch_output_idx in range(len(outputs))
                    ]
                )
                .mean()
                .item()
            )

            # Add to sacred
            # self.log(
            #     f"train_{log_key}", avg_batch_log, self.current_epoch
            # )
            if isinstance(self.logger, list):
                self.logger[0].log_metric(
                    f"train_{log_key}", avg_batch_log, self.current_epoch
                )
            else:
                self.logger.log_metric(
                    f"train_{log_key}", avg_batch_log, self.current_epoch
                )

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, unused: Optional[int] = 0) -> None:
        names = ["image", "sound"]
        for k, processor in zip(names, self.model.processors):
            for name, p in processor.named_parameters():
                if p.grad is not None:
                    norm = p.grad.data.norm(2)
                    self.log(f"grad_norm_{k}/{name}", norm, on_step=True)

    def validation_step(self, batch, batch_idx):

        # if self.model_name in ("gmc4", "gmc5", "gmc6", "gmc7", "gmc8", "gmc9", "gmc10"):
        if "gmc" in self.model_name and self.model_name != "gmc":
            data = batch[0], batch[1]
        else: data = batch[0]
        output_dict = self.model.validation_step(data, self.train_config)
        return output_dict

    def validation_epoch_end(self, outputs):
        # print(outputs)
        log_keys = list(outputs[0].keys())
        for log_key in log_keys:
            avg_batch_log = (
                torch.stack(
                    [
                        outputs[batch_output_idx][log_key]
                        for batch_output_idx in range(len(outputs))
                    ]
                )
                .mean()
                .item()
            )
            self.log(f"val_{log_key}", avg_batch_log, on_epoch=True, logger=False)
            if isinstance(self.logger, list):
                self.logger[0].log_metric(
                    f"val_{log_key}", avg_batch_log, self.current_epoch
                )
            else:
                self.logger.log_metric(
                    f"val_{log_key}", avg_batch_log, self.current_epoch
                )
            print(log_key, avg_batch_log)
            # self.logger.log_metric(f"val_{log_key}", avg_batch_log, self.current_epoch)
