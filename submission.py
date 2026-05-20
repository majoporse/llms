from __future__ import annotations

import platform
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from urllib.request import urlopen

LEIPZIG_URL = "https://downloads.wortschatz-leipzig.de/corpora/eng_news_2025_1M.tar.gz"
DEVEL_URL = "https://is.muni.cz/el/fi/jaro2026/PV026/um/data/devel.tsv"
REPO_URL = "https://github.com/majoporse/llms.git"


def _repo_root() -> Path:
    return Path.cwd() / "llms"


def _run(command: list[str], cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=str(cwd) if cwd else None, check=True)


def _venv_python(repo_root: Path) -> Path:
    return repo_root / ".venv" / "bin" / "python"


def _ensure_repo_clone() -> Path:
    repo_root = _repo_root()
    if (repo_root / ".git").exists():
        return repo_root

    repo_root.parent.mkdir(parents=True, exist_ok=True)
    if repo_root.exists():
        shutil.rmtree(repo_root)
    _run(["git", "clone", REPO_URL, str(repo_root)])
    return repo_root


def _ensure_uv(repo_root: Path) -> Path:
    """Downloads the official standalone uv binary based on OS and architecture."""
    uv_bin = repo_root / ("uv.exe" if platform.system() == "Windows" else "uv")
    if uv_bin.exists():
        return uv_bin

    # Map system architecture to official uv release targets
    sys_os = platform.system().lower()
    arch = platform.machine().lower()
    
    if "x86_64" in arch or "amd64" in arch:
        arch_target = "x86_64"
    elif "arm64" in arch or "aarch64" in arch:
        arch_target = "aarch64"
    else:
        raise RuntimeError(f"Unsupported architecture: {arch}")

    if sys_os == "linux":
        target = f"{arch_target}-unknown-linux-musl"
        archive_ext = ".tar.gz"
    elif sys_os == "darwin":
        target = f"{arch_target}-apple-darwin"
        archive_ext = ".tar.gz"
    elif sys_os == "windows":
        target = f"{arch_target}-pc-windows-msvc"
        archive_ext = ".zip"
    else:
        raise RuntimeError(f"Unsupported OS: {sys_os}")

    url = f"https://github.com/astral-sh/uv/releases/latest/download/uv-{target}{archive_ext}"
    archive_path = repo_root / f"uv_download{archive_ext}"

    print(f"Downloading uv binary from {url}...")
    _download_file(url, archive_path)

    # Extract the binary from the archive
    if archive_ext == ".tar.gz":
        with tarfile.open(archive_path, "r:gz") as tar:
            for member in tar.getmembers():
                if member.name.endswith("/uv"):
                    with tar.extractfile(member) as src, open(uv_bin, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    break
    else:  # .zip
        with zipfile.ZipFile(archive_path) as zip_ref:
            for member in zip_ref.namelist():
                if member.endswith("/uv.exe"):
                    with zip_ref.open(member) as src, open(uv_bin, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    break

    archive_path.unlink()
    uv_bin.chmod(0o755)  # Mark executable
    return uv_bin


def _ensure_venv(repo_root: Path) -> Path:
    venv_python = _venv_python(repo_root)
    if venv_python.exists():
        return venv_python

    # Fetch uv executable automatically
    uv_path = _ensure_uv(repo_root)

    print("Syncing project dependencies via uv...")
    _run([str(uv_path), "sync", "--extra", "gpu", "--no-dev"], cwd=repo_root)
    
    return venv_python


def _download_file(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest

    tmp_dest = dest.with_suffix(dest.suffix + ".tmp")
    with urlopen(url) as response, open(tmp_dest, "wb") as out:
        shutil.copyfileobj(response, out)
    tmp_dest.replace(dest)
    return dest


def prepare_training_data() -> None:
    repo_root = _ensure_repo_clone()
    _ = _ensure_venv(repo_root)
    data_dir = repo_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    archive_path = data_dir / "eng_news_2025_1M.tar.gz"
    devel_path = data_dir / "devel.tsv"
    sentences_filename = "eng_news_2025_1M-sentences.txt"

    print("Downloading Leipzig corpus...")
    _download_file(LEIPZIG_URL, archive_path)

    print("Extracting corpus...")
    with tarfile.open(archive_path, "r:gz") as tar:
        name = f"eng_news_2025_1M/{sentences_filename}"
        for member in tar.getmembers():
            if member.name == name:
                tar.extract(member, data_dir)
                break
        else:
            raise FileNotFoundError(f"Sentence file {name} not found in the archive")
    # copy the sentences file to the data directory
    sentences_path = data_dir / "eng_news_2025_1M" / sentences_filename
    shutil.copy(sentences_path, data_dir / sentences_filename)
    

    print("Downloading devel data...")
    _download_file(DEVEL_URL, devel_path)


def train_model() -> None:
    repo_root = _ensure_repo_clone()
    venv_python = _ensure_venv(repo_root)
    _run([str(venv_python), "-m", "scripts.tok_train"], cwd=repo_root)
    _run([str(venv_python), "-m", "scripts.base_train"], cwd=repo_root)
    _run([str(venv_python), "-m", "scripts.grammar_preference_tune"], cwd=repo_root)


def evaluate_model(filename: str) -> None:
    repo_root = _ensure_repo_clone()
    input_path = Path(filename).resolve()
    cwd = Path.cwd()
    output_path = cwd / "eval.output"
    repo_output = repo_root / "eval.output"

    venv_python = _ensure_venv(repo_root)
    _run(
        [str(venv_python), "-m", "scripts.grammar_eval", "--tsv-path", str(input_path)],
        cwd=repo_root,
    )

    shutil.copyfile(repo_output, output_path)


if __name__ == "__main__":
    # prepare_training_data()
    # train_model()
    evaluate_model("data/eval-input.tsv")
    # evaluate_model("llms/data/devel.tsv")