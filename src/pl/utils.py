import sys

from pytorch_lightning.callbacks import ProgressBar
from tqdm import tqdm


class CustomProgressBar(ProgressBar):

    def init_validation_tqdm(self) -> tqdm:
        """ Override this to customize the tqdm bar for validation. """
        bar = tqdm(
            desc='Validating',
            position=(2 * self.process_position + 1),
            disable=True,
            leave=False,
            dynamic_ncols=True,
            file=sys.stdout
        )
        return bar