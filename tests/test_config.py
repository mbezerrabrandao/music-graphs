from pathlib import Path

from music_graphs.config import load_config


def test_load_paper_config() -> None:
    config = load_config(Path("configs/paper.yaml"))
    assert config["parameters"]["sessions"]["threshold_minutes"] == 60
    assert config["parameters"]["node2vec"]["p"] == 2.0
    assert config["parameters"]["node2vec"]["q"] == 2.0
