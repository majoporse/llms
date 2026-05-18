from pathlib import Path
import importlib.util
import sys

import torch


def load_module():
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    path = root / "scripts/grammar_preference_tune.py"
    spec = importlib.util.spec_from_file_location("grammar_preference_tune", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


gptune = load_module()


def test_read_pairs_parses_two_column_tsv(tmp_path: Path):
    path = tmp_path / "pairs.tsv"
    path.write_text("good one\tbad one\nokay\tnope\n", encoding="utf-8")

    pairs = gptune.read_pairs(path)

    assert pairs == [("good one", "bad one"), ("okay", "nope")]


def test_split_pairs_is_deterministic():
    pairs = [(f"good{i}", f"bad{i}") for i in range(10)]

    train_a, test_a = gptune.split_pairs(pairs, test_fraction=0.2, seed=123)
    train_b, test_b = gptune.split_pairs(pairs, test_fraction=0.2, seed=123)

    assert train_a == train_b
    assert test_a == test_b
    assert len(test_a) == 2
    assert len(train_a) == 8
    assert set(train_a).isdisjoint(set(test_a))


def test_split_pairs_uses_entire_input():
    pairs = [(f"good{i}", f"bad{i}") for i in range(5)]

    train, test = gptune.split_pairs(pairs, test_fraction=0.4, seed=0)

    assert sorted(train + test) == sorted(pairs)


def test_pairwise_preference_loss_prefers_good_sentence():
    good_nll = torch.tensor(1.0)
    bad_nll = torch.tensor(3.0)

    loss = gptune.pairwise_preference_loss(good_nll, bad_nll)

    assert loss.item() < 0.15


def test_pairwise_preference_loss_penalizes_wrong_order():
    good_nll = torch.tensor(3.0)
    bad_nll = torch.tensor(1.0)

    loss = gptune.pairwise_preference_loss(good_nll, bad_nll)

    assert loss.item() > 1.5


def test_batch_sentence_nll_matches_mean_loss_shape():
    class DummyTokenizer:
        def get_bos_token_id(self):
            return 99

        def __call__(self, sentence, prepend=None):
            mapping = {
                "a": [99, 1, 2],
                "bb": [99, 3, 4, 5],
            }
            return mapping[sentence]

    class DummyModel:
        def get_device(self):
            return torch.device("cpu")

        def __call__(self, ids, targets, loss_reduction="none"):
            return torch.tensor(1.0)

    nll = gptune.batch_sentence_nll(DummyModel(), DummyTokenizer(), ["a", "bb"])

    assert nll.shape == (2,)
    assert torch.allclose(nll, torch.tensor([1.0, 1.0]))


def test_script_imports_via_runpy():
    import runpy

    module = runpy.run_path(str(Path("scripts/grammar_preference_tune.py")))

    assert "pairwise_preference_loss" in module


def test_settings_dataclass_exists():
    settings = gptune.GrammarPreferenceTuneSettings()

    assert settings.pairs_path.endswith("devel.tsv")
    assert settings.test_fraction == 0.05


def test_save_model_matches_base_train_checkpoint_layout(tmp_path: Path):
    class DummyTokenizer:
        pass

    class DummyConfig:
        def __init__(self):
            self.sequence_len = 128
            self.vocab_size = 1000

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
            self.config = DummyConfig()

    settings = gptune.GrammarPreferenceTuneSettings()
    gptune.save_model(DummyModel(), DummyTokenizer(), settings, "nanochat", tmp_path, 7)

    assert (tmp_path / "model_000007.pt").exists()
    assert (tmp_path / "meta_000007.json").exists()


def test_save_model_includes_model_config(tmp_path: Path):
    class DummyTokenizer:
        pass

    class DummyConfig:
        def __init__(self):
            self.sequence_len = 128
            self.vocab_size = 1000

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
            self.config = DummyConfig()

    settings = gptune.GrammarPreferenceTuneSettings()
    gptune.save_model(DummyModel(), DummyTokenizer(), settings, "nanochat", tmp_path, 9)

    meta = (tmp_path / "meta_000009.json").read_text(encoding="utf-8")

    assert '"model_config"' in meta


def test_score_pair_batch_keeps_gradients():
    class DummyTokenizer:
        def get_bos_token_id(self):
            return 99

        def __call__(self, sentence, prepend=None):
            mapping = {
                "good": [99, 1, 2],
                "bad": [99, 3, 4],
            }
            return mapping[sentence]

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def get_device(self):
            return torch.device("cpu")

        def __call__(self, ids, targets, loss_reduction="mean"):
            return self.weight * ids.sum().float()

    model = DummyModel()
    loss, correct, count = gptune.score_pair_batch(model, DummyTokenizer(), [("good", "bad")])

    assert loss.requires_grad
    loss.backward()
    assert model.weight.grad is not None
    assert correct == 1
    assert count == 1
