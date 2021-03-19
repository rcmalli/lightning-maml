from typing import Any, Dict, Sequence, Tuple, Union

import higher
import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.optim import Optimizer
import torch.nn.functional as F


class BaseModel(pl.LightningModule):
    def __init__(self, cfg: DictConfig, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.save_hyperparameters(cfg)

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Method for the forward pass.
        'training_step', 'validation_step' and 'test_step' should call
        this method in order to compute the output predictions and the loss.
        Returns:
            output_dict: forward output containing the predictions (output logits ecc...) and the loss if any.
        """
        raise NotImplementedError

    def step(self, train: bool, batch: Any):
        raise NotImplementedError

    def training_step(self, batch: Any, batch_idx: int):
        outer_loss, inner_loss, outer_acc, inner_acc = self.step(True, batch)

        self.log_dict(
            {"metatrain/inner_loss": inner_loss.item(),
             "metatrain/inner_accuracy": inner_acc.compute()},
            on_epoch=False,
            on_step=True,
            prog_bar=False
        )
        self.log_dict(
            {"metatrain/outer_loss": outer_loss.item(),
             "metatrain/outer_accuracy": outer_acc.compute()},
            on_epoch=False,
            on_step=True,
            prog_bar=True
        )

    def validation_step(self, batch: Any, batch_idx: int):
        torch.set_grad_enabled(True)
        self.cnn.train()
        outer_loss, inner_loss, outer_acc, inner_acc = self.step(False, batch)
        self.log_dict(
            {"metaval/inner_loss": inner_loss.item(),
             "metaval/inner_accuracy": inner_acc.compute()},
            prog_bar=False
        )
        self.log_dict(
            {"metaval/outer_loss": outer_loss.item(),
             "metaval/outer_accuracy": outer_acc.compute()},
            prog_bar=True
        )

    def test_step(self, batch: Any, batch_idx: int):
        torch.set_grad_enabled(True)
        self.cnn.train()
        outer_loss, inner_loss, outer_acc, inner_acc = self.step(False, batch)
        self.log_dict(
            {"metatest/outer_loss": outer_loss.item(),
             "metatest/inner_loss": inner_loss.item(),
             "metatest/inner_accuracy": inner_acc.compute(),
             "metatest/outer_accuracy": outer_acc.compute()},

        )


class MAMLModel(BaseModel):

    def __init__(self, torch_module, cfg: DictConfig, *args, **kwargs) -> None:
        super().__init__(cfg=cfg, *args, **kwargs)
        self.cnn = hydra.utils.instantiate(torch_module,
                                           num_classes=cfg.data.datamodule.nway)
        self.cnn = self.cnn.to(device=self.device)
        self.inner_optimizer = hydra.utils.instantiate(
            cfg.optim.inner_optimizer, params=self.cnn.parameters())
        self.cfg = cfg
        self.save_hyperparameters(cfg)

        metric = pl.metrics.Accuracy()
        self.train_inner_accuracy = metric.clone()
        self.train_outer_accuracy = metric.clone()
        self.val_inner_accuracy = metric.clone()
        self.val_outer_accuracy = metric.clone()
        self.test_inner_accuracy = metric.clone()
        self.test_outer_accuracy = metric.clone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cnn(x)

    def step(self, train: bool, batch: Any):
        self.cnn.zero_grad()
        outer_optimizer = self.optimizers()
        train_inputs, train_targets = batch['support']
        test_inputs, test_targets = batch['query']

        train_inputs = train_inputs.to(device=self.device)
        train_targets = train_targets.to(device=self.device)
        test_inputs = test_inputs.to(device=self.device)
        test_targets = test_targets.to(device=self.device)

        metric = pl.metrics.Accuracy()
        outer_loss = torch.tensor(0., device=self.device)
        inner_loss = torch.tensor(0., device='cpu')
        outer_accuracy = metric.clone()
        inner_accuracy = metric.clone()
        for task_idx, (train_input, train_target, test_input,
                       test_target) in enumerate(
            zip(train_inputs, train_targets,
                test_inputs, test_targets)):
            track_higher_grads = True if train else False
            with higher.innerloop_ctx(self.cnn, self.inner_optimizer,
                                      copy_initial_weights=False,
                                      track_higher_grads=track_higher_grads) as (
                    fmodel, diffopt):
                for k in range(self.cfg.data.datamodule.num_inner_steps):
                    train_logit = fmodel(train_input)
                    loss = F.cross_entropy(train_logit, train_target)
                    diffopt.step(loss)

                with torch.no_grad():
                    train_logit = fmodel(train_input)
                    train_preds = torch.softmax(train_logit, dim=-1)
                    inner_loss += F.cross_entropy(train_logit,
                                                  train_target).cpu()
                    inner_accuracy.update(train_preds.cpu(), train_target.cpu())

                test_logit = fmodel(test_input)
                outer_loss += F.cross_entropy(test_logit, test_target)
                with torch.no_grad():
                    test_preds = torch.softmax(train_logit, dim=-1)
                    outer_accuracy.update(test_preds.cpu(), test_target.cpu())

        if train:
            self.manual_backward(outer_loss, outer_optimizer)
            outer_optimizer.step()

        outer_loss.div_(task_idx + 1)
        inner_loss.div_(task_idx + 1)

        return outer_loss, inner_loss, outer_accuracy, inner_accuracy

    def configure_optimizers(
            self,
    ) -> Union[Optimizer, Tuple[Sequence[Optimizer], Sequence[Any]]]:
        outer_optimizer = hydra.utils.instantiate(
            self.cfg.optim.outer_optimizer, params=self.parameters()
        )

        if self.cfg.optim.use_lr_scheduler:
            scheduler = hydra.utils.instantiate(
                self.cfg.optim.lr_scheduler, optimizer=outer_optimizer
            )
            return [outer_optimizer], [scheduler]

        return outer_optimizer
