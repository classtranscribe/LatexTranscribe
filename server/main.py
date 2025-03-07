import argparse
import os
import yaml
import warnings

from src.pipeline import Pipeline


def parse_args():
    def dir_or_file_path(path):
        if os.path.exists(path):
            return path
        else:
            raise argparse.ArgumentTypeError(f"{path} is not a valid path")

    parser = argparse.ArgumentParser(
        description="Run a task with a given configuration file."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/config.yaml",
        help="Path to the main configuration file.",
    )
    parser.add_argument(
        "--input_path",
        type=dir_or_file_path,
        default="inputs/",
        help="Path to the input directory / file.",
    )
    parser.add_argument(
        "--output_path",
        type=dir_or_file_path,
        default="../../outputs/",
        help="Path to the output directory.",
    )
    return parser.parse_args()


def load_config(config_path):
    if config_path is None:
        warnings.warn(
            "Configuration path is None. Please provide a valid configuration file path. "
        )
        return None

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    print(args.input_path, args.output_path)
    pipeline = Pipeline(config, args.input_path, args.output_path)
    pipeline.predict(save=True)
