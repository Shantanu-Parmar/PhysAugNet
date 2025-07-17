# physaug/utils/config.py
import yaml
import os
import yaml
import argparse
def load_config(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_config(config, path):
    with open(path, 'w') as f:
        yaml.dump(config, f)




def load_config_with_overrides(default_path="config/default.yml"):
    # Load defaults from YAML
    with open(default_path, "r") as f:
        config = yaml.safe_load(f)

    # Parse CLI args
    parser = argparse.ArgumentParser()
    for key, val in config.items():
        arg_type = type(val) if val is not None else str
        parser.add_argument(f"--{key}", type=arg_type, default=val)
    args = parser.parse_args()

    # Override config with CLI
    config.update(vars(args))
    return config
