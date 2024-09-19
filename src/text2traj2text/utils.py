"""utils

Author: Hikaru Asano
Affiliation: CyberAgent AI Lab / The University of Tokyo
"""

import os

from omegaconf import OmegaConf


def save_config(config: dict, dirname: str) -> None:
    """save configiration of training for logging

    Args:
        config (dict): configuration of training
        dirname (str): save directory name
    """

    os.makedirs(dirname, exist_ok=True)
    with open(f"{dirname}/config.yaml", "w") as f:
        OmegaConf.save(config, f)
