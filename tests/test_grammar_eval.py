from pathlib import Path
import importlib.util
import sys


def load_module():
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    path = root / "scripts/grammar_eval.py"
    spec = importlib.util.spec_from_file_location("grammar_eval", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


gval = load_module()


def test_settings_dataclass_exists():
    settings = gval.GrammarEvalSettings()

    assert settings.tsv_path.endswith("devel.tsv")
    assert settings.device_type == "cuda"
    assert settings.batch_size == 16


def test_score_pair_batch_counts_correct_predictions():
    class DummyTokenizer:
        def get_bos_token_id(self):
            return 99

        def __call__(self, sentence, prepend=None):
            mapping = {
                "good": [99, 1, 2],
                "bad": [99, 3, 4],
                "also good": [99, 5, 6],
                "also bad": [99, 7, 8],
            }
            return mapping[sentence]

    class DummyModel:
        def get_device(self):
            return "cpu"

    calls = []

    def fake_score_pair(model, tokenizer, sent_a, sent_b):
        calls.append((sent_a, sent_b))
        return 0.0, 1.0, sent_a in {"good", "also good"}

    original = gval.score_pair
    gval.score_pair = fake_score_pair
    try:
        correct, total, predictions = gval.score_pair_batch(
            DummyModel(),
            DummyTokenizer(),
            [("good", "bad"), ("also good", "also bad")],
        )
    finally:
        gval.score_pair = original

    assert correct == 2
    assert total == 2
    assert predictions == [True, True]
    assert calls == [("good", "bad"), ("also good", "also bad")]


def test_main_accepts_settings(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "pairs.tsv").write_text("a\tb\n", encoding="utf-8")

    settings = gval.GrammarEvalSettings(tsv_path=str(tmp_path / "pairs.tsv"), batch_size=1)

    monkeypatch.setattr(gval, "load_model_and_tokenizer", lambda settings, device: (None, None, "dummy"))
    monkeypatch.setattr(gval, "compute_init", lambda device_type: (False, 0, 0, 1, None))
    monkeypatch.setattr(gval, "compute_cleanup", lambda: None)
    monkeypatch.setattr(gval, "print0", lambda *args, **kwargs: None)
    monkeypatch.setattr(gval, "read_pairs", lambda path: [])

    gval.grammar_eval(settings)
