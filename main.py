import slideflow.slide.backends.vips as vips_backend

# Wrap the existing detect_mpp to fall back to 0.25 when it returns None
_orig_detect = vips_backend.detect_mpp
def detect_with_default(path, loaded_image=None):
    mpp = _orig_detect(path, loaded_image)
    return mpp if mpp is not None else 0.25

# Monkey-patch it in place
vips_backend.detect_mpp = detect_with_default

import sys
import logging
import argparse
import torch

from pathbench.experiment import Experiment

# 1) grab the root logger and reset handlers
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.handlers.clear()

# 2) create a single formatter
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 3) DEBUG → stderr only
debug_handler = logging.StreamHandler(sys.stderr)
debug_handler.setLevel(logging.DEBUG)
# filter so that *only* DEBUG records go here
debug_handler.addFilter(lambda record: record.levelno == logging.DEBUG)
debug_handler.setFormatter(formatter)
logger.addHandler(debug_handler)

# 4) INFO and above → stdout
info_handler = logging.StreamHandler(sys.stdout)
info_handler.setLevel(logging.INFO)
# no extra filter: all INFO+ will pass
info_handler.setFormatter(formatter)
logger.addHandler(info_handler)

# 5) (optional) also write all DEBUG+ to a file
file_handler = logging.FileHandler('debug_logfile.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Silence overly‐chatty libraries if you like
logging.getLogger("VIPS").setLevel(logging.WARNING)

def main(config_path):
    """"
    Main function to run the experiment
    """
    # Create an instance of the Experiment class
    experiment = Experiment(config_path)
    experiment.run()

if __name__ == "__main__":
    # Add arguments
    parser = argparse.ArgumentParser(description='Run PathDev experiment')
    parser.add_argument('--config', type=str, default='conf.yaml', help='Path to the configuration file')
    args = parser.parse_args()

    main(args.config)