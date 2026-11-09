"""
Local & Hybrid AI Pipeline v2.4 for 声幕 SubCast:
1. VAD: Silero-VAD (natural pause detection & speech segmentation)
2. ASR (Primary): Alibaba Qwen3-ASR-0.6B (30 languages + 22 dialects int8, native punctuation & ITN)
3. ASR (Fallback): SenseVoice-Small / sherpa-onnx Streaming Zipformer + Edge-Punct-Casing
4. Translation (Primary): Context-Aware External LLM API (MiniMax / DeepSeek / OpenAI-compatible)
   with a 6-sentence sliding context window to auto-correct homophone ASR errors and produce natural TV subtitles.
5. Translation (Fallback): Offline CTranslate2 (int8) + SentencePiece when no API key is configured.
"""

from __future__ import annotations

import collections
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Deque, List, Optional, Tuple

import ctranslate2
import httpx
import numpy as np
import sentencepiece as spm
import sherpa_onnx

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"


@dataclass
class SubtitleEvent:
    channel: str
    en: str
    zh: str
    is_final: bool
    timestamp: float
    latency_ms: float = 0.0
    engine_info: str = ""
    pipeline_elapsed_ms: float = 0.0


@dataclass
class LLMConfig:
    enabled: bool = True
    base_url: str = "https://api.minimaxi.com/v1"
    api_key: str = ""
    model: str = "MiniMax-M3.1-Flash"
    source_lang: str = "auto"  # "auto" | "en" | "ja" | "ko" | "yue" | "zh"


class LocalTranslator:
    """Offline English -> Chinese neural machine translation powered by CTranslate2."""

    def __init__(self, nmt_dir: Optional[Path] = None, num_threads: int = 4) -> None:
        self.nmt_dir = nmt_dir or (MODELS_DIR / "nmt")
        model_path = self.nmt_dir / "model"
        sp_path = self.nmt_dir / "sentencepiece.model"
        if not model_path.exists() or not sp_path.exists():
            raise FileNotFoundError(
                f"NMT model not found in {self.nmt_dir}. Run `python scripts/download_models.py` first."
            )

        self.sp = spm.SentencePieceProcessor(model_file=str(sp_path))
        self.translator = ctranslate2.Translator(
            str(model_path),
            device="cpu",
            compute_type="int8",
            inter_threads=1,
            intra_threads=num_threads,
        )
        self._cache: dict[str, str] = {}

    def translate(self, text: str) -> str:
        clean = text.strip()
        if not clean:
            return ""
        if clean in self._cache:
            return self._cache[clean]

        tokens = self.sp.encode(clean, out_type=str)
        if not tokens:
            return ""

        results = self.translator.translate_batch(
            [tokens],
            beam_size=2,
            max_decoding_length=128,
            repetition_penalty=1.15,
        )
        out_tokens = results[0].hypotheses[0]
        zh_text = self.sp.decode(out_tokens).replace("▁", " ").strip()

        if len(self._cache) > 512:
            self._cache.clear()
        self._cache[clean] = zh_text
        return zh_text


class LLMContextTranslator:
    """
    Context-aware live TV subtitle translator using any OpenAI-compatible Chat Completions API.
    Keeps a persistent HTTP connection pool and a 4-sentence sliding context window so the LLM
    can automatically fix homophone ASR errors and maintain natural terminology across sentences.
    """

    SYSTEM_PROMPT = (
        "你是一位顶级的国际电视直播同声传译员。\n"
        "任务：将用户提供的【当前待译原句】进行整句理解与语序重组，翻译为自然、地道、逻辑连贯的简体中文电视字幕。\n"
        "核心要求：\n"
        "1. 整句意译与语序重组：【当前待译原句】包含完整的一句或多句连续直播语音。请先通读整段话，按照中文母语的叙事逻辑重新组织语序（如将英文后置定语、状语从句自然前置或顺译），将零散分句融合为通顺完整的中文整句，切忌按英文语序逐词死板切割。\n"
        "2. 智能纠错与术语还原：结合【频道】与【前文语境】，自动判断并修正前端语音识别（ASR）可能产生的同音错词、断句粘连、人名、地名、机构名或财经/体育/新闻专业术语。\n"
        "3. 严格只输出【当前待译原句】对应的简体中文译文，绝不输出任何解释、前缀、引号或多余内容。"
    )

    def __init__(self, fallback_translator: LocalTranslator) -> None:
        self.fallback = fallback_translator
        self.config = LLMConfig()
        self.context_window: Deque[Tuple[str, str]] = collections.deque(maxlen=6)
        self.last_error: str = ""
        self.last_used_engine: str = "CTranslate2 (离线兜底)"
        # Persistent connection pool for minimal TLS handshake overhead
        self._http = httpx.Client(timeout=6.5, verify=False)
        self._cache: dict[str, str] = {}

    def clear_context(self) -> None:
        self.context_window.clear()

    def update_config(
        self,
        enabled: Optional[bool] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        source_lang: Optional[str] = None,
    ) -> None:
        if enabled is not None:
            self.config.enabled = enabled
        if base_url is not None and base_url.strip():
            self.config.base_url = base_url.strip().rstrip("/")
        if api_key is not None:
            self.config.api_key = api_key.strip()
        if model is not None and model.strip():
            self.config.model = model.strip()
        if source_lang is not None and source_lang.strip():
            self.config.source_lang = source_lang.strip()
        self._cache.clear()

    @property
    def is_llm_ready(self) -> bool:
        return bool(self.config.enabled and self.config.api_key and self.config.base_url)

    def translate(self, text: str, channel_name: str = "Live TV", is_final: bool = True) -> str:
        clean = text.strip()
        if not clean:
            return ""

        # If input is already predominantly Chinese, return as-is
        cjk_chars = sum(1 for c in clean if "\u4e00" <= c <= "\u9fff")
        if cjk_chars > len(clean) * 0.55:
            return clean

        if not self.is_llm_ready:
            self.last_used_engine = "CTranslate2 (未配置API Key，使用离线小模型)"
            return self.fallback.translate(clean)

        # For interim previews, check cache first
        if clean in self._cache:
            return self._cache[clean]

        try:
            zh_text = self._call_llm_api(clean, channel_name)
            if zh_text:
                self.last_error = ""
                self.last_used_engine = f"LLM ({self.config.model})"
                if len(self._cache) > 256:
                    self._cache.clear()
                self._cache[clean] = zh_text
                if is_final:
                    self.context_window.append((clean, zh_text))
                return zh_text
        except Exception as exc:
            self.last_error = str(exc)
            print(f"[LLMTranslator] API fallback due to error: {exc}")

        self.last_used_engine = "CTranslate2 (API异常自动降级)"
        return self.fallback.translate(clean)

    def _call_llm_api(self, source_text: str, channel_name: str) -> str:
        base = self.config.base_url.rstrip("/")
        is_minimax = any(d in base.lower() for d in ("minimaxi.com", "minimax.io", "minimax.chat"))
        is_anthropic = "/anthropic" in base.lower()

        context_lines = []
        for prev_src, prev_zh in self.context_window:
            context_lines.append(f"- 原文：{prev_src}\n  译文：{prev_zh}")
        context_block = "\n".join(context_lines) if context_lines else "（无前文）"

        user_prompt = (
            f"【频道】：{channel_name}\n"
            f"【前文语境】：\n{context_block}\n\n"
            f"【当前待译原句】：\n{source_text}"
        )

        if is_anthropic:
            url = base if base.endswith("/messages") else (
                f"{base}/messages" if base.endswith("/v1") else f"{base}/v1/messages"
            )
            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "x-api-key": self.config.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self.config.model,
                "max_tokens": 1024,
                "temperature": 0.2,
                "system": self.SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": user_prompt}],
            }
            resp = self._http.post(url, headers=headers, json=payload)
            if resp.status_code >= 400:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            data = resp.json()
            text_blocks = [
                b.get("text", "")
                for b in data.get("content", [])
                if isinstance(b, dict) and b.get("type") == "text"
            ]
            content = "\n".join(text_blocks).strip()
        else:
            url = base if base.endswith("/chat/completions") else f"{base}/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self.config.model,
                "messages": [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.2,
                "max_tokens": 1024 if is_minimax else 200,
                "stream": False,
            }
            if is_minimax:
                # Separate MiniMax M-series reasoning into reasoning_details so content is pure translation
                payload["reasoning_split"] = True
                if "m3" in self.config.model.lower():
                    payload["reasoning_effort"] = "low"

            resp = self._http.post(url, headers=headers, json=payload)
            if resp.status_code >= 400:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            data = resp.json()

            # Check MiniMax-specific base_resp status_code
            base_resp = data.get("base_resp")
            if isinstance(base_resp, dict) and base_resp.get("status_code", 0) not in (0, None):
                raise RuntimeError(
                    f"MiniMax API Error {base_resp.get('status_code')}: {base_resp.get('status_msg')}"
                )

            choices = data.get("choices") or []
            if not choices:
                raise RuntimeError(f"Empty choices in response: {str(data)[:200]}")
            content = (choices[0].get("message", {}).get("content") or "").strip()

        # Strip <think>...</think> blocks if any reasoning model includes them in content
        content = re.sub(r"<think>[\s\S]*?</think>", "", content, flags=re.IGNORECASE).strip()
        content = re.sub(r"<think>[\s\S]*$", "", content, flags=re.IGNORECASE).strip()
        # Strip accidental prefix or wrapping quotes
        content = re.sub(r"^(译文[:：]\s*)", "", content).strip().strip('"\'“”')
        return content


class StreamingSpeechPipeline:
    """
    Upgraded v2 Speech-to-Subtitle Pipeline:
    - Primary ASR: Silero-VAD + Alibaba SenseVoice-Small (int8, multilingual, noise-robust, native punctuation)
    - Primary Translation: LLMContextTranslator (with CTranslate2 offline fallback)
    """

    _SENSEVOICE_TAG_RE = re.compile(r"<\|[^|>]+\|>")
    _LAST_WORD_RE = re.compile(
        r"([A-Za-z\u00C0-\u017F]+(?:'[A-Za-z\u00C0-\u017F]+)?)[^A-Za-z\u00C0-\u017F]*$"
    )

    # Words that cannot validly end a complete standalone sentence (English / French / Spanish / German).
    # If ASR outputs a segment ending with one of these (even with a trailing '.'),
    # the speaker paused mid-sentence and we must merge with the next segment.
    _DANGLING_END_WORDS = {
        # English Conjunctions, Relative Pronouns, Prepositions, Determiners, Auxiliaries
        "and", "or", "but", "nor", "so", "yet", "for", "because", "although", "though",
        "while", "whilst", "whereas", "if", "unless", "until", "till", "since", "when",
        "whenever", "where", "wherever", "whether", "before", "after", "once",
        "that", "which", "who", "whom", "whose", "what", "whatever", "whichever", "whoever",
        "to", "of", "in", "on", "at", "by", "with", "from", "into", "onto", "upon",
        "about", "above", "across", "against", "along", "amid", "among", "around", "as",
        "behind", "below", "beneath", "beside", "between", "beyond", "concerning", "considering",
        "despite", "down", "during", "except", "following", "including", "inside", "like",
        "near", "off", "opposite", "out", "outside", "over", "past", "regarding", "round",
        "through", "throughout", "toward", "towards", "under", "underneath", "unlike",
        "up", "via", "within", "without",
        "the", "a", "an", "this", "these", "those", "my", "your", "his", "her", "its",
        "our", "their", "some", "any", "every", "each", "no", "another", "such",
        "is", "are", "was", "were", "am", "be", "been", "being",
        "have", "has", "had", "having", "do", "does", "did",
        "will", "would", "shall", "should", "can", "could", "may", "might", "must",
        "isn't", "aren't", "wasn't", "weren't", "haven't", "hasn't", "hadn't",
        "won't", "wouldn't", "don't", "doesn't", "didn't", "can't", "couldn't", "shouldn't",
        "not", "also", "very", "too", "quite", "rather", "extremely", "particularly",
        "especially", "nearly", "almost", "just", "only", "even", "still",
        "mr", "mrs", "ms", "dr", "prof", "president", "minister", "senator", "general",
        # French (Français) Conjunctions, Prepositions, Articles, Pronouns, Auxiliaries
        "et", "ou", "mais", "donc", "car", "ni", "que", "qui", "quoi", "dont", "où",
        "quand", "comme", "si", "lorsque", "puisque", "quoique", "parce", "pendant",
        "avant", "après", "depuis", "pour", "dans", "en", "sur", "sous", "avec", "sans",
        "chez", "vers", "contre", "entre", "par", "de", "du", "des", "au", "aux", "à",
        "le", "la", "les", "un", "une", "ce", "cet", "cette", "ces", "mon", "ton", "son",
        "ma", "ta", "sa", "mes", "tes", "ses", "notre", "votre", "leur", "leurs",
        "est", "sont", "était", "étaient", "sera", "seront", "ont", "avait", "avaient",
        "va", "vont", "peut", "peuvent", "doit", "doivent", "très", "trop", "aussi",
        # Spanish (Español) Conjunctions, Prepositions, Articles, Auxiliaries
        "y", "e", "u", "pero", "sino", "porque", "aunque", "como", "cuando", "donde",
        "mientras", "quien", "cual", "cuyo", "del", "al", "con", "para", "sin", "sobre",
        "hacia", "hasta", "desde", "según", "contra", "el", "los", "las", "unos", "unas",
        "este", "esta", "estos", "estas", "mi", "tu", "su", "nuestro", "son", "era",
        "eran", "fue", "fueron", "ha", "han", "había", "van", "puede", "pueden", "muy",
        # German (Deutsch) Conjunctions, Prepositions, Articles, Auxiliaries
        "und", "oder", "aber", "denn", "sondern", "dass", "weil", "obwohl", "wenn",
        "als", "während", "bevor", "nachdem", "seit", "bis", "ob", "wie", "wo",
        "im", "am", "auf", "für", "mit", "von", "vom", "zu", "zum", "zur", "bei", "beim",
        "nach", "aus", "über", "unter", "vor", "hinter", "zwischen", "durch", "gegen",
        "ohne", "um", "der", "die", "das", "den", "dem", "ein", "eine", "einen", "einem",
        "einer", "eines", "mein", "dein", "sein", "ihr", "unser", "ist", "sind", "war",
        "waren", "hat", "haben", "wird", "werden", "kann", "können", "muss", "müssen",
    }

    _PROPER_INIT_WORDS = {
        "I", "I'm", "I've", "I'll", "I'd", "US", "U.S.", "UK", "U.K.", "UN", "EU",
        "NATO", "BBC", "CNN", "CNBC", "NHK", "GDP", "AI", "CEO", "FBI", "CIA", "WHO",
        "WTO", "IMF", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    }

    def __init__(
        self,
        models_dir: Optional[Path] = None,
        on_subtitle: Optional[Callable[[SubtitleEvent], None]] = None,
        num_threads: int = 4,
    ) -> None:
        self.models_dir = models_dir or MODELS_DIR
        self.on_subtitle = on_subtitle
        self.channel_name = "Live Stream"
        self.num_threads = num_threads
        # "full_sentence" (9-28 words, default) | "paragraph" (18-38 words) | "fast_clause" (5-16 words)
        self.sentence_mode: str = "full_sentence"
        self.async_translate_mode: bool = False

        self.local_translator = LocalTranslator(self.models_dir / "nmt", num_threads=num_threads)
        self.llm_translator = LLMContextTranslator(fallback_translator=self.local_translator)

        # Punctuator is still kept for direct plain-text testing or Zipformer fallback
        punct_dir = self.models_dir / "punct"
        punct_model_cfg = sherpa_onnx.OnlinePunctuationModelConfig(
            cnn_bilstm=str(punct_dir / "model.int8.onnx"),
            bpe_vocab=str(punct_dir / "bpe.vocab"),
            num_threads=2,
            provider="cpu",
        )
        self.punctuator = sherpa_onnx.OnlinePunctuation(
            sherpa_onnx.OnlinePunctuationConfig(model_config=punct_model_cfg)
        )

        # Primary Unified ASR: Qwen3-ASR-0.6B-int8 (30 languages + 22 dialects, zero-config auto LID & punctuation)
        self.vad_model_path = self.models_dir / "vad" / "silero_vad.onnx"
        self.qwen3_dir = self.models_dir / "qwen3_asr"
        self.sensevoice_dir = self.models_dir / "sensevoice"
        self.use_qwen3_asr = (
            self.vad_model_path.exists()
            and (self.qwen3_dir / "conv_frontend.onnx").exists()
            and (self.qwen3_dir / "encoder.int8.onnx").exists()
            and (self.qwen3_dir / "decoder.int8.onnx").exists()
            and (self.qwen3_dir / "tokenizer" / "vocab.json").exists()
        )
        self.use_sensevoice = (
            self.vad_model_path.exists()
            and (self.sensevoice_dir / "model.int8.onnx").exists()
            and (self.sensevoice_dir / "tokens.txt").exists()
        )

        self.qwen3_recognizer: Optional[sherpa_onnx.OfflineRecognizer] = None
        self.sensevoice_recognizer: Optional[sherpa_onnx.OfflineRecognizer] = None
        self.vad: Optional[sherpa_onnx.VoiceActivityDetector] = None
        self.last_asr_engine: str = "Qwen3-ASR-0.6B" if self.use_qwen3_asr else "SenseVoice-Small"

        if self.use_qwen3_asr:
            self._init_qwen3_asr()
            self._init_vad()
            self.use_sensevoice = True  # Enables VAD-based full-sentence streaming pipeline
            print("[AIPipeline] Loaded Silero-VAD + Qwen3-ASR-0.6B-int8 (30语种+22方言全自动统一识别).")
        elif self.use_sensevoice:
            self._init_sensevoice("auto")
            self._init_vad()
            print("[AIPipeline] Loaded Silero-VAD + SenseVoice-Small (fallback).")
        else:
            print("[AIPipeline] Offline VAD/ASR not found, falling back to Zipformer.")

        # Also keep Zipformer ready if present on disk
        asr_dir = self.models_dir / "asr"
        self.zipformer_recognizer: Optional[sherpa_onnx.OnlineRecognizer] = None
        if (asr_dir / "tokens.txt").exists() and (asr_dir / "encoder.int8.onnx").exists():
            self.zipformer_recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
                tokens=str(asr_dir / "tokens.txt"),
                encoder=str(asr_dir / "encoder.int8.onnx"),
                decoder=str(asr_dir / "decoder.int8.onnx"),
                joiner=str(asr_dir / "joiner.int8.onnx"),
                num_threads=num_threads,
                sample_rate=16000,
                feature_dim=80,
                enable_endpoint_detection=True,
                rule1_min_trailing_silence=2.0,
                rule2_min_trailing_silence=0.9,
                rule3_min_utterance_length=16.0,
                decoding_method="greedy_search",
                provider="cpu",
            )

        self._vad_remainder = np.zeros(0, dtype=np.float32)
        self._speech_accum = np.zeros(0, dtype=np.float32)
        self._pending_clauses: List[str] = []
        self._pending_first_ts: float = 0.0
        self._pending_last_asr_ms: float = 480.0
        self._samples_since_vad_pop: int = 0
        self._last_interim_emit_ts = 0.0
        self._last_interim_en = ""
        self.reset_stream()

    def _init_qwen3_asr(self) -> None:
        qwen_threads = min(8, max(4, os.cpu_count() or 6))
        self.qwen3_recognizer = sherpa_onnx.OfflineRecognizer.from_qwen3_asr(
            conv_frontend=str(self.qwen3_dir / "conv_frontend.onnx"),
            encoder=str(self.qwen3_dir / "encoder.int8.onnx"),
            decoder=str(self.qwen3_dir / "decoder.int8.onnx"),
            tokenizer=str(self.qwen3_dir / "tokenizer"),
            num_threads=qwen_threads,
            sample_rate=16000,
            feature_dim=128,
            decoding_method="greedy_search",
            provider="cpu",
        )

    def _init_sensevoice(self, lang: str = "auto") -> None:
        sv_lang = "" if lang == "auto" else lang
        self.sensevoice_recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=str(self.sensevoice_dir / "model.int8.onnx"),
            tokens=str(self.sensevoice_dir / "tokens.txt"),
            num_threads=self.num_threads,
            sample_rate=16000,
            use_itn=True,
            language=sv_lang,
            provider="cpu",
        )

    def _init_vad(self) -> None:
        silero_cfg = sherpa_onnx.SileroVadModelConfig(
            model=str(self.vad_model_path),
            threshold=0.38,
            min_silence_duration=0.48,
            min_speech_duration=0.25,
            window_size=512,
            max_speech_duration=8.5,
        )
        vad_cfg = sherpa_onnx.VadModelConfig(
            silero_vad=silero_cfg,
            sample_rate=16000,
            num_threads=1,
            provider="cpu",
        )
        self.vad = sherpa_onnx.VoiceActivityDetector(vad_cfg, buffer_size_in_seconds=60)

    def reset_stream(self, channel_name: Optional[str] = None) -> None:
        if channel_name:
            if channel_name != self.channel_name:
                self.llm_translator.clear_context()
            self.channel_name = channel_name
        if self.vad is not None:
            self.vad.reset()
        self._vad_remainder = np.zeros(0, dtype=np.float32)
        self._speech_accum = np.zeros(0, dtype=np.float32)
        self._pending_clauses = []
        self._pending_first_ts = 0.0
        self._pending_last_asr_ms = 480.0
        self._samples_since_vad_pop = 0
        self._last_interim_emit_ts = 0.0
        self._last_interim_en = ""
        self.stream = self.zipformer_recognizer.create_stream() if self.zipformer_recognizer else None

    def punctuate(self, raw_text: str, ensure_end_punct: bool = False) -> str:
        clean = " ".join(raw_text.strip().lower().split())
        if not clean:
            return ""
        res = self.punctuator.add_punctuation_with_case(clean).strip()
        if ensure_end_punct and res and res[-1] not in ".?!":
            res += "."
        return res

    def _recognize_sensevoice(self, samples: np.ndarray, sample_rate: int = 16000) -> str:
        if len(samples) < 2400:
            return ""

        t0 = time.perf_counter()
        if self.qwen3_recognizer is not None:
            stream = self.qwen3_recognizer.create_stream()
            stream.accept_waveform(sample_rate, samples)
            self.qwen3_recognizer.decode_stream(stream)
            raw = stream.result.text.strip()
            self._pending_last_asr_ms = (time.perf_counter() - t0) * 1000.0
            self.last_asr_engine = "Qwen3-ASR-0.6B"
            return raw

        if self.sensevoice_recognizer is None:
            return ""

        stream = self.sensevoice_recognizer.create_stream()
        stream.accept_waveform(sample_rate, samples)
        self.sensevoice_recognizer.decode_stream(stream)
        raw = stream.result.text.strip()
        self._pending_last_asr_ms = (time.perf_counter() - t0) * 1000.0
        cleaned = self._SENSEVOICE_TAG_RE.sub("", raw).strip()
        cleaned = re.sub(r"\.\s+([a-z])", r", \1", cleaned)
        self.last_asr_engine = "SenseVoice"
        return cleaned

    @classmethod
    def _ends_with_dangling_word(cls, text: str) -> bool:
        m = cls._LAST_WORD_RE.search(text.strip())
        if not m:
            return False
        last_word = m.group(1).lower()
        return last_word in cls._DANGLING_END_WORDS

    @staticmethod
    def _count_words_or_units(text: str) -> int:
        cjk = sum(1 for c in text if "\u3040" <= c <= "\u9fff" or "\uac00" <= c <= "\ud7af")
        latin_words = len(
            re.findall(r"[A-Za-z0-9\u00C0-\u017F]+(?:'[A-Za-z0-9\u00C0-\u017F]+)?", text)
        )
        return latin_words + int(round(cjk / 1.6))

    def _get_mode_thresholds(self) -> Tuple[int, int, float, float]:
        """Returns (min_words, max_words, post_vad_silence_flush_sec, max_wait_sec)."""
        if self.sentence_mode == "paragraph":
            return (18, 38, 0.75, 8.5)
        if self.sentence_mode == "fast_clause":
            return (5, 16, 0.35, 3.2)
        # Default: "full_sentence"
        return (9, 28, 0.48, 5.5)

    def _append_clause_to_pending(self, raw_clause: str, now: float) -> None:
        clause = " ".join(raw_clause.strip().split())
        if not clause:
            return

        if not self._pending_clauses:
            self._pending_first_ts = now

        # Check if previous buffered clause ended WITHOUT terminal punctuation (i.e. mid-sentence continuation)
        # OR if the new clause starts with a lowercase letter (meaning ASR itself detected continuation)
        if self._pending_clauses:
            prev = self._pending_clauses[-1]
            if clause[0].islower() and prev.endswith((".", "。")):
                prev = prev[:-1].rstrip()
                self._pending_clauses[-1] = prev
            prev_ended_sentence = bool(prev) and prev[-1] in ".?!。？！"
            if not prev_ended_sentence:
                # Lowercase false sentence-initial capitalization from ASR on continuation segment
                tokens = clause.split(" ", 1)
                first_tok = tokens[0]
                if (
                    len(first_tok) > 1
                    and first_tok[0].isupper()
                    and first_tok[1:].islower()
                    and first_tok.rstrip(",.?!;:") not in self._PROPER_INIT_WORDS
                ):
                    tokens[0] = first_tok[0].lower() + first_tok[1:]
                    clause = " ".join(tokens)

        # If this clause ends on a dangling word (e.g. "announced that." or "going to."),
        # strip the false trailing period added by ASR so it merges seamlessly with the next segment.
        if self._ends_with_dangling_word(clause) and clause[-1] in ".。":
            clause = clause[:-1].rstrip()

        self._pending_clauses.append(clause)

    def _get_combined_pending_text(self) -> str:
        if not self._pending_clauses:
            return ""
        return " ".join(self._pending_clauses).strip()

    def _is_pending_grammatically_complete(self) -> bool:
        combined = self._get_combined_pending_text()
        if not combined:
            return False
        if self._ends_with_dangling_word(combined):
            return False
        return combined[-1] in ".?!。？！"

    def _should_flush_pending(
        self, now: float, is_currently_speaking: bool, post_vad_silence_sec: float
    ) -> bool:
        if not self._pending_clauses:
            return False

        combined = self._get_combined_pending_text()
        word_count = self._count_words_or_units(combined)
        is_complete = self._is_pending_grammatically_complete()
        min_words, max_words, silence_flush_sec, max_wait_sec = self._get_mode_thresholds()

        # 1. Hard cap on total words or waiting duration so subtitles never fall too far behind
        if word_count >= max_words or (now - self._pending_first_ts) >= max_wait_sec:
            return True

        # 2. Target full-sentence length reached and grammatically complete
        if is_complete and word_count >= min_words:
            return True

        # 3. Speaker stopped speaking (silence timeout after VAD segment end)
        if not is_currently_speaking:
            if is_complete and post_vad_silence_sec >= silence_flush_sec:
                return True
            if not is_complete and post_vad_silence_sec >= (silence_flush_sec + 0.85):
                return True

        return False

    def _flush_pending_to_event(self) -> Optional[SubtitleEvent]:
        combined = self._get_combined_pending_text()
        first_ts = self._pending_first_ts
        post_silence_ms = (self._samples_since_vad_pop / 16000.0) * 1000.0
        self._pending_clauses = []
        self._pending_first_ts = 0.0
        self._samples_since_vad_pop = 0
        if not combined or len(combined.strip(" .,?!。，？！")) < 2:
            return None

        if combined[-1] not in ".?!。？！":
            combined += "."

        now_ts = time.time()
        agg_span_ms = max(0.0, (now_ts - first_ts) * 1000.0) if first_ts > 0 else 0.0
        pre_llm_elapsed_ms = self._pending_last_asr_ms + post_silence_ms + min(1200.0, agg_span_ms * 0.35)

        if self.async_translate_mode:
            # Return immediately so FFmpeg audio reader & VAD never block on LLM HTTP latency
            return SubtitleEvent(
                channel=self.channel_name,
                en=combined,
                zh="",
                is_final=True,
                timestamp=now_ts,
                latency_ms=0.0,
                engine_info=self.last_asr_engine,
                pipeline_elapsed_ms=round(pre_llm_elapsed_ms, 1),
            )

        t0 = time.perf_counter()
        zh_text = self.llm_translator.translate(
            combined, channel_name=self.channel_name, is_final=True
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        pipeline_elapsed_ms = pre_llm_elapsed_ms + latency_ms

        ev = SubtitleEvent(
            channel=self.channel_name,
            en=combined,
            zh=zh_text,
            is_final=True,
            timestamp=time.time(),
            latency_ms=round(latency_ms, 1),
            engine_info=f"{self.last_asr_engine} + {self.llm_translator.last_used_engine}",
            pipeline_elapsed_ms=round(pipeline_elapsed_ms, 1),
        )
        if self.on_subtitle:
            self.on_subtitle(ev)
        return ev

    def complete_event_translation(self, ev: SubtitleEvent) -> SubtitleEvent:
        """Completes LLM translation on a background worker thread without blocking audio capture."""
        asr_tag = (
            ev.engine_info
            if ev.engine_info and "+" not in ev.engine_info
            else self.last_asr_engine
        )
        t0 = time.perf_counter()
        zh_text = self.llm_translator.translate(
            ev.en, channel_name=ev.channel or self.channel_name, is_final=ev.is_final
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        ev.zh = zh_text
        ev.latency_ms = round(latency_ms, 1)
        ev.timestamp = time.time()
        ev.engine_info = f"{asr_tag} + {self.llm_translator.last_used_engine}"
        ev.pipeline_elapsed_ms = round(ev.pipeline_elapsed_ms + latency_ms, 1)
        if self.on_subtitle:
            self.on_subtitle(ev)
        return ev

    def process_text_direct(self, raw_text: str, is_final: bool = True) -> Optional[SubtitleEvent]:
        t0 = time.perf_counter()
        # If text is all-lowercase or all-uppercase without punctuation, run punctuator first
        has_punct = any(p in raw_text for p in (".", ",", "?", "!", "。", "？", "！"))
        if not has_punct and raw_text.isascii():
            en_text = self.punctuate(raw_text, ensure_end_punct=is_final)
        else:
            en_text = raw_text.strip()
        if not en_text:
            return None

        zh_text = self.llm_translator.translate(
            en_text, channel_name=self.channel_name, is_final=is_final
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        event = SubtitleEvent(
            channel=self.channel_name,
            en=en_text,
            zh=zh_text,
            is_final=is_final,
            timestamp=time.time(),
            latency_ms=round(latency_ms, 1),
            engine_info=f"{self.last_asr_engine} + {self.llm_translator.last_used_engine}",
        )
        if self.on_subtitle:
            self.on_subtitle(event)
        return event

    def feed_pcm_s16le(self, pcm_bytes: bytes) -> List[SubtitleEvent]:
        if not pcm_bytes:
            return []
        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        return self.feed_samples(samples)

    def feed_samples(self, samples: np.ndarray, sample_rate: int = 16000) -> List[SubtitleEvent]:
        if not self.use_sensevoice or self.vad is None:
            return self._feed_samples_zipformer(samples, sample_rate)

        events: List[SubtitleEvent] = []
        buf = np.concatenate([self._vad_remainder, samples])
        window_size = 512
        idx = 0
        now = time.time()

        while idx + window_size <= len(buf):
            win = buf[idx : idx + window_size]
            idx += window_size
            self.vad.accept_waveform(win)
            speaking_now = self.vad.is_speech_detected()

            if speaking_now:
                self._samples_since_vad_pop = 0
                self._speech_accum = np.concatenate([self._speech_accum, win])
                if len(self._speech_accum) > 16000 * 12:
                    self._speech_accum = self._speech_accum[-16000 * 12 :]
            elif self._pending_clauses:
                self._samples_since_vad_pop += window_size

            # 1. Check if Silero-VAD completed one or more speech segments
            while not self.vad.empty():
                seg_samples = np.array(self.vad.front.samples, dtype=np.float32)
                self.vad.pop()
                self._speech_accum = np.zeros(0, dtype=np.float32)
                self._last_interim_en = ""
                self._samples_since_vad_pop = 0

                if len(seg_samples) < int(16000 * 0.35):
                    continue

                src_text = self._recognize_sensevoice(seg_samples)
                if not src_text or len(src_text.strip(" .,?!。，？！")) < 2:
                    continue

                self._append_clause_to_pending(src_text, now)
                if self._should_flush_pending(
                    now=now,
                    is_currently_speaking=self.vad.is_speech_detected(),
                    post_vad_silence_sec=0.0,
                ):
                    ev = self._flush_pending_to_event()
                    if ev is not None:
                        events.append(ev)

            # 2. If we have buffered clauses waiting for continuation, check if silence timeout expired
            if self._pending_clauses and not speaking_now:
                post_silence_sec = self._samples_since_vad_pop / float(sample_rate)
                if self._should_flush_pending(
                    now=now,
                    is_currently_speaking=False,
                    post_vad_silence_sec=post_silence_sec,
                ):
                    ev = self._flush_pending_to_event()
                    if ev is not None:
                        events.append(ev)

        self._vad_remainder = buf[idx:]
        return events

    def flush(self) -> List[SubtitleEvent]:
        events: List[SubtitleEvent] = []
        if self.use_sensevoice and self.vad is not None:
            # Feed 0.8s of silence to flush VAD
            silence = np.zeros(int(16000 * 0.8), dtype=np.float32)
            events.extend(self.feed_samples(silence))
            self.vad.flush()
            now = time.time()
            while not self.vad.empty():
                seg_samples = np.array(self.vad.front.samples, dtype=np.float32)
                self.vad.pop()
                src_text = self._recognize_sensevoice(seg_samples)
                if src_text and len(src_text.strip(" .,?!。，？！")) >= 2:
                    self._append_clause_to_pending(src_text, now)
            if self._pending_clauses:
                ev = self._flush_pending_to_event()
                if ev is not None:
                    events.append(ev)
            self.reset_stream()
            return events

        return events

    def _feed_samples_zipformer(
        self, samples: np.ndarray, sample_rate: int = 16000
    ) -> List[SubtitleEvent]:
        events: List[SubtitleEvent] = []
        self.stream.accept_waveform(sample_rate, samples)
        while self.zipformer_recognizer.is_ready(self.stream):
            self.zipformer_recognizer.decode_stream(self.stream)
        raw_result = self.zipformer_recognizer.get_result(self.stream).strip()
        is_endpoint = self.zipformer_recognizer.is_endpoint(self.stream)
        if is_endpoint and raw_result:
            ev = self.process_text_direct(raw_result, is_final=True)
            if ev:
                events.append(ev)
            self.zipformer_recognizer.reset(self.stream)
        return events
