import yaml
from pathlib import Path

def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

def out(cfg, *parts):
    """Return path under cfg['paths']['output'] / parts."""
    return Path(cfg["paths"]["output"]).joinpath(*parts)

def fig(cfg, *parts):
    """Return path under cfg['paths']['figures'] / parts."""
    return Path(cfg["paths"]["figures"]).joinpath(*parts)
