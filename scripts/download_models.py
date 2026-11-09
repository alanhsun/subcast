"""
Download and prepare local offline AI models for LinguaRelay-TV (声桥 TV 版):
1. VAD: Silero-VAD (silero_vad.onnx, ~630KB)
2. ASR (Primary): Alibaba SenseVoice-Small (zh/en/ja/ko/yue multilingual int8, ~228MB)
3. ASR (Fallback): sherpa-onnx-streaming-zipformer-en-2023-06-26 (int8)
4. Punctuation: sherpa-onnx-online-punct-en-2024-08-06 (Edge-Punct-Casing int8)
5. Translation (Fallback Offline NMT): CTranslate2 en->zh offline model + SentencePiece
"""

from __future__ import annotations

import gzip
import os
import shutil
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data"
SEED_DATA_DIR = ROOT_DIR / "seed_data"

VAD_DIR = MODELS_DIR / "vad"
QWEN3_ASR_DIR = MODELS_DIR / "qwen3_asr"
SENSEVOICE_DIR = MODELS_DIR / "sensevoice"
CANARY_DIR = MODELS_DIR / "canary"
ASR_DIR = MODELS_DIR / "asr"
PUNCT_DIR = MODELS_DIR / "punct"
NMT_DIR = MODELS_DIR / "nmt"

SILERO_VAD_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"

HF_QWEN3_ASR_BASE = (
    "https://huggingface.co/csukuangfj2/"
    "sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/resolve/main"
)

QWEN3_ASR_FILES = {
    "conv_frontend.onnx": f"{HF_QWEN3_ASR_BASE}/conv_frontend.onnx",
    "encoder.int8.onnx": f"{HF_QWEN3_ASR_BASE}/encoder.int8.onnx",
    "decoder.int8.onnx": f"{HF_QWEN3_ASR_BASE}/decoder.int8.onnx",
    "tokenizer/merges.txt": f"{HF_QWEN3_ASR_BASE}/tokenizer/merges.txt",
    "tokenizer/tokenizer_config.json": f"{HF_QWEN3_ASR_BASE}/tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json": f"{HF_QWEN3_ASR_BASE}/tokenizer/vocab.json",
}

HF_SENSEVOICE_BASE = (
    "https://huggingface.co/csukuangfj/"
    "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/resolve/main"
)

SENSEVOICE_FILES = {
    "model.int8.onnx": f"{HF_SENSEVOICE_BASE}/model.int8.onnx",
    "tokens.txt": f"{HF_SENSEVOICE_BASE}/tokens.txt",
    "test_en.wav": f"{HF_SENSEVOICE_BASE}/test_wavs/en.wav",
}

HF_CANARY_BASE = (
    "https://huggingface.co/csukuangfj/"
    "sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8/resolve/main"
)

CANARY_FILES = {
    "encoder.int8.onnx": f"{HF_CANARY_BASE}/encoder.int8.onnx",
    "decoder.int8.onnx": f"{HF_CANARY_BASE}/decoder.int8.onnx",
    "tokens.txt": f"{HF_CANARY_BASE}/tokens.txt",
}

HF_ASR_BASE = (
    "https://huggingface.co/csukuangfj/"
    "sherpa-onnx-streaming-zipformer-en-2023-06-26/resolve/main"
)

ASR_FILES = {
    "encoder.int8.onnx": f"{HF_ASR_BASE}/encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
    "decoder.int8.onnx": f"{HF_ASR_BASE}/decoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
    "joiner.int8.onnx": f"{HF_ASR_BASE}/joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
    "tokens.txt": f"{HF_ASR_BASE}/tokens.txt",
    "bpe.model": f"{HF_ASR_BASE}/bpe.model",
    "test_en.wav": f"{HF_ASR_BASE}/test_wavs/0.wav",
}

PUNCT_TAR_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "punctuation-models/sherpa-onnx-online-punct-en-2024-08-06.tar.bz2"
)

ARGOS_EN_ZH_URL = "https://argos-net.com/v1/translate-en_zh-1_9.argosmodel"


def _build_candidate_urls(url: str) -> list[str]:
    """Generate primary + Mainland China mirror fallback URLs for HuggingFace & GitHub Releases."""
    hf_endpoint = os.environ.get("HF_ENDPOINT", "").strip().rstrip("/")
    candidates: list[str] = []
    if "huggingface.co/" in url:
        if hf_endpoint and hf_endpoint != "https://huggingface.co":
            candidates.append(url.replace("https://huggingface.co", hf_endpoint))
        candidates.append(url)
        mirror_url = url.replace("https://huggingface.co", "https://hf-mirror.com")
        if mirror_url not in candidates:
            candidates.append(mirror_url)
    elif "github.com/" in url:
        candidates.append(url)
        candidates.append(f"https://ghfast.top/{url}")
    else:
        candidates.append(url)
    return candidates


def download_file(url: str, dest: Path, desc: str = "") -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_dest = dest.with_suffix(dest.suffix + ".part")
    label = desc or dest.name
    candidates = _build_candidate_urls(url)
    last_err: Exception | None = None

    for cand_url in candidates:
        print(f"[Download] {label} <- {cand_url}")
        try:
            req = urllib.request.Request(
                cand_url,
                headers={"User-Agent": "LinguaRelay-TV/2.4"},
            )
            with urllib.request.urlopen(req, timeout=180) as resp, open(tmp_dest, "wb") as out:
                total = int(resp.headers.get("Content-Length", 0))
                downloaded = 0
                chunk_size = 512 * 1024
                last_pct = -1
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    out.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = int(downloaded * 100 / total)
                        if pct >= last_pct + 10 or downloaded == total:
                            last_pct = pct
                            mb_done = downloaded / (1024 * 1024)
                            mb_total = total / (1024 * 1024)
                            print(f"  -> {label}: {pct}% ({mb_done:.1f} / {mb_total:.1f} MB)")
            tmp_dest.replace(dest)
            return
        except Exception as exc:
            last_err = exc
            print(f"  [Warn] Mirror failed ({exc}), trying next fallback...")
            tmp_dest.unlink(missing_ok=True)

    if last_err:
        raise last_err


def ensure_seed_data() -> None:
    """Populate /app/data with initial playlist & EPG if mounted volume on Unraid NAS is empty."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not SEED_DATA_DIR.exists():
        return
    seed_pl = SEED_DATA_DIR / "playlist_source.txt"
    dst_pl = DATA_DIR / "playlist_source.txt"
    if seed_pl.exists() and not dst_pl.exists():
        shutil.copy2(seed_pl, dst_pl)
        print("[Init] Seeded default 337-channel IPTV playlist into data/playlist_source.txt")

    seed_epg_gz = SEED_DATA_DIR / "epg.xml.gz"
    dst_epg_gz = DATA_DIR / "epg.xml.gz"
    dst_epg_xml = DATA_DIR / "epg.xml"
    if seed_epg_gz.exists():
        if not dst_epg_gz.exists():
            shutil.copy2(seed_epg_gz, dst_epg_gz)
            print("[Init] Seeded EPG archive into data/epg.xml.gz")
        if not dst_epg_xml.exists() and dst_epg_gz.exists():
            with gzip.open(dst_epg_gz, "rb") as f_in, open(dst_epg_xml, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            print("[Init] Decompressed data/epg.xml.gz -> data/epg.xml")


def ensure_vad_model() -> None:
    VAD_DIR.mkdir(parents=True, exist_ok=True)
    target = VAD_DIR / "silero_vad.onnx"
    if target.exists() and target.stat().st_size > 10000:
        print(f"[OK] Silero-VAD ready: {target.name}")
        return
    download_file(SILERO_VAD_URL, target, desc="Silero-VAD/silero_vad.onnx")


def ensure_qwen3_asr_models() -> None:
    (QWEN3_ASR_DIR / "tokenizer").mkdir(parents=True, exist_ok=True)
    for rel_path, url in QWEN3_ASR_FILES.items():
        target = QWEN3_ASR_DIR / rel_path
        if target.exists() and target.stat().st_size > 100:
            print(f"[OK] Qwen3-ASR-0.6B (30语种统一识别) file ready: {rel_path}")
            continue
        download_file(url, target, desc=f"Qwen3-ASR-0.6B/{rel_path}")


def ensure_sensevoice_models() -> None:
    SENSEVOICE_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in SENSEVOICE_FILES.items():
        target = SENSEVOICE_DIR / filename
        if target.exists() and target.stat().st_size > 100:
            print(f"[OK] SenseVoice file ready: {target.name}")
            continue
        download_file(url, target, desc=f"SenseVoice/{filename}")


def ensure_canary_models() -> None:
    CANARY_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in CANARY_FILES.items():
        target = CANARY_DIR / filename
        if target.exists() and target.stat().st_size > 100:
            print(f"[OK] Canary-180M-Flash (en/fr/de/es) file ready: {target.name}")
            continue
        download_file(url, target, desc=f"Canary-180M-Flash/{filename}")


def ensure_asr_models() -> None:
    ASR_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in ASR_FILES.items():
        target = ASR_DIR / filename
        if target.exists() and target.stat().st_size > 100:
            print(f"[OK] Zipformer file ready: {target.name}")
            continue
        download_file(url, target, desc=f"ASR/{filename}")


def ensure_punct_models() -> None:
    PUNCT_DIR.mkdir(parents=True, exist_ok=True)
    model_int8 = PUNCT_DIR / "model.int8.onnx"
    bpe_vocab = PUNCT_DIR / "bpe.vocab"
    if model_int8.exists() and bpe_vocab.exists():
        print("[OK] Punctuation model (Edge-Punct-Casing) ready.")
        return

    archive_path = MODELS_DIR / "punct.tar.bz2"
    if not archive_path.exists():
        download_file(PUNCT_TAR_URL, archive_path, desc="Edge-Punct-Casing tar.bz2")

    extract_tmp = MODELS_DIR / "_punct_tmp"
    if extract_tmp.exists():
        shutil.rmtree(extract_tmp)
    extract_tmp.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive_path, "r:bz2") as tar:
        tar.extractall(extract_tmp)

    inner_dir = extract_tmp / "sherpa-onnx-online-punct-en-2024-08-06"
    for item in inner_dir.iterdir():
        if item.is_file():
            shutil.copy2(item, PUNCT_DIR / item.name)

    shutil.rmtree(extract_tmp, ignore_errors=True)
    archive_path.unlink(missing_ok=True)


def ensure_nmt_models() -> None:
    NMT_DIR.mkdir(parents=True, exist_ok=True)
    ct2_model_bin = NMT_DIR / "model" / "model.bin"
    sp_model = NMT_DIR / "sentencepiece.model"
    if ct2_model_bin.exists() and sp_model.exists():
        print("[OK] CTranslate2 English->Chinese NMT model ready.")
        return

    archive_path = MODELS_DIR / "translate-en_zh.argosmodel"
    if not archive_path.exists():
        download_file(ARGOS_EN_ZH_URL, archive_path, desc="CTranslate2 en->zh NMT")

    extract_tmp = MODELS_DIR / "_nmt_tmp"
    if extract_tmp.exists():
        shutil.rmtree(extract_tmp)
    extract_tmp.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(extract_tmp)

    pkg_dirs = [p for p in extract_tmp.iterdir() if p.is_dir()]
    pkg_dir = pkg_dirs[0]
    for item in pkg_dir.iterdir():
        dest = NMT_DIR / item.name
        if dest.exists():
            if dest.is_dir():
                shutil.rmtree(dest)
            else:
                dest.unlink()
        shutil.move(str(item), str(dest))

    shutil.rmtree(extract_tmp, ignore_errors=True)
    archive_path.unlink(missing_ok=True)


def main() -> int:
    print("=== LinguaRelay-TV v2.4 Offline Model & Seed Data Initializer ===")
    ensure_seed_data()
    ensure_vad_model()
    ensure_qwen3_asr_models()
    if "--legacy" in sys.argv:
        ensure_sensevoice_models()
        ensure_asr_models()
    ensure_punct_models()
    ensure_nmt_models()
    print("=== All models are ready in:", MODELS_DIR, "===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
