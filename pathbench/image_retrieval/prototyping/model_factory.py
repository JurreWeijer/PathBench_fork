import os
from .model_PANTHER import PANTHER
from dataclasses import dataclass, asdict
from typing import Optional, Union, Callable
import json
import logging

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger(__name__)

CONFIG_DIR = "/exports/path-pulmogroep-hpc/Jurre/SalvDataset-CBIR/PathBench_fork/pathbench/image_retrieval/prototyping/configs"

@dataclass
class PretrainedConfig:
    def to_json_file(self, json_file_path: Union[str, os.PathLike]):
        """
        Save this instance to a JSON file.
        Args:
            json_file_path: Path to the JSON file in which this configuration instance's parameters will be saved.
        """
        config_dict = {k: v for k, v in asdict(self).items()}
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(json.dumps(
                config_dict, indent=2, sort_keys=False) + "\n")

    @classmethod
    def from_pretrained(cls, config_path: Union[str, os.PathLike], update_dict={}):
        config_dict = json.load(open(config_path))
        for key in update_dict:
            if key in config_dict:
                config_dict[key] = update_dict[key]
        config = cls(**config_dict)
        return config

    def save_pretrained(self, save_directory: Union[str, os.PathLike]):
        """
        Save a configuration object to the directory `save_directory`, so that it can be re-loaded using the
        [`~PretrainedConfig.from_pretrained`] class method.
        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the configuration JSON file will be saved (will be created if it does not exist).
        """
        if os.path.isfile(save_directory):
            raise AssertionError(
                f"Provided path ({save_directory}) should be a directory, not a file")

        os.makedirs(save_directory, exist_ok=True)

        # If we save using the predefined names, we can load using `from_pretrained`
        output_config_file = os.path.join(save_directory, "config.json")

        self.to_json_file(output_config_file)
        logger.info(f"Configuration saved in {output_config_file}")

@dataclass
class PANTHERConfig(PretrainedConfig):
    in_dim: int = 768
    n_classes: int = 2
    heads: int = 1
    em_iter: int = 3
    tau: float = 0.001
    embed_dim: int = 512
    ot_eps: int = 0.1
    n_fc_layers: int = 1
    dropout: float = 0.
    out_type: str = 'param_cat'
    out_size: int = 3
    load_proto: bool = True
    proto_path: str = '.'
    fix_proto: bool = True

def create_prototyping_model(model_config, in_dim, n_proto, load_proto, fix_proto, proto_path, out_type):
    """
    Create classification or survival models
    """
    config_path = os.path.join(CONFIG_DIR, model_config, 'config.json')
    assert os.path.exists(config_path), f"Config path {config_path} doesn't exist!"

    update_dict = {'in_dim': in_dim,
                   'out_size': n_proto,
                   'load_proto': load_proto,
                   'fix_proto': fix_proto,
                   'proto_path': proto_path}


    update_dict.update({'out_type': out_type})
    config = PANTHERConfig.from_pretrained(config_path, update_dict=update_dict)
    model = PANTHER(config=config, mode="emb")

    return model