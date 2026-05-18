from pathlib import Path
import importlib.util
import sys


def load_module():
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    path = root / "scripts/base_train_config.py"
    spec = importlib.util.spec_from_file_location("base_train_config", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_settings_from_env_reads_training_values(monkeypatch):
    monkeypatch.setenv("NANOCHAT_RUN", "exp1")
    monkeypatch.setenv("NANOCHAT_DEPTH", "24")
    monkeypatch.setenv("NANOCHAT_DEVICE_BATCH_SIZE", "16")
    monkeypatch.setenv("NANOCHAT_FP8", "1")
    monkeypatch.setenv("NANOCHAT_MODEL_TAG", "d24")

    module = load_module()
    settings = module.settings_from_env()

    assert settings.run == "exp1"
    assert settings.depth == 24
    assert settings.device_batch_size == 16
    assert settings.fp8 is True
    assert settings.model_tag == "d24"


def test_settings_from_env_uses_defaults(monkeypatch):
    module = load_module()
    settings = module.settings_from_env()

    assert settings.depth == 20
    assert settings.device_batch_size == 32
    assert settings.target_param_data_ratio == 12
