import argparse
import yaml
import warnings
from src.pipeline import Pipeline

def parse_args():
    parser = argparse.ArgumentParser(description="Run a task with a given configuration file.")
    parser.add_argument('--config', type=str, default="./configs/config.yaml", help='Path to the configuration files.')
    parser.add_argument('--output_dir', type=str, default="outputs/pipeline", help='Path to the configuration files.')
    return parser.parse_args()

def load_config(config_path):
    if config_path is None:
        warnings.warn(
            ("Configuration path is None. Please provide a valid configuration file path. ")
        )
        return None
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    pipeline = Pipeline(config)
    pipeline.predict()

