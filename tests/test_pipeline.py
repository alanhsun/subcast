from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from server.ai_pipeline import StreamingSpeechPipeline

ROOT_DIR = Path(__file__).resolve().parent.parent
TEST_WAV = next(
    (
        p
        for p in (
            ROOT_DIR / "models" / "qwen3_asr" / "test_en.wav",
            ROOT_DIR / "models" / "sensevoice" / "test_en.wav",
            ROOT_DIR / "models" / "asr" / "test_en.wav",
        )
        if p.exists()
    ),
    ROOT_DIR / "models" / "qwen3_asr" / "test_en.wav",
)


def test_streaming_speech_pipeline_end_to_end() -> None:
    emitted = []
    pipeline = StreamingSpeechPipeline(on_subtitle=lambda ev: emitted.append(ev))
    pipeline.reset_stream(channel_name="BBC News Test")

    assert TEST_WAV.exists(), f"Missing test wav: {TEST_WAV}"
    with wave.open(str(TEST_WAV), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        sr = wf.getframerate()
        assert sr == 16000
        # Feed in 100ms chunks (1600 samples = 3200 bytes) to simulate live streaming
        chunk_bytes = 3200
        while True:
            data = wf.readframes(chunk_bytes // 2)
            if not data:
                break
            pipeline.feed_pcm_s16le(data)

    pipeline.flush()

    assert len(emitted) > 0, "Expected at least one subtitle event from streaming wav"
    finals = [e for e in emitted if e.is_final]
    assert len(finals) > 0, "Expected at least one final subtitle event"

    for ev in emitted:
        print(
            f"[{'FINAL' if ev.is_final else 'LIVE '}] ({ev.latency_ms}ms) "
            f"EN: {ev.en} | ZH: {ev.zh}"
        )

    full_en = " ".join(e.en for e in finals)
    full_zh = " ".join(e.zh for e in finals)
    assert len(full_en) > 10
    assert len(full_zh) > 4
    # Verify punctuation & casing were restored
    assert any(ch.isupper() for ch in full_en)
    assert any(p in full_en for p in (".", ",", "?", "!"))


def test_full_sentence_clause_aggregation_and_dangling_merge() -> None:
    pipeline = StreamingSpeechPipeline()
    pipeline.reset_stream(channel_name="BBC World News")
    pipeline.sentence_mode = "full_sentence"

    now = 1000.0
    # 1. First VAD segment ends with a dangling word "that." (false period from ASR)
    pipeline._append_clause_to_pending("In Washington today, the Prime Minister announced that.", now)
    assert not pipeline._is_pending_grammatically_complete()
    assert not pipeline._should_flush_pending(
        now=now, is_currently_speaking=False, post_vad_silence_sec=0.3
    )

    # 2. Second VAD segment continues the sentence with a false initial capital "The"
    pipeline._append_clause_to_pending(
        "The central bank will keep interest rates unchanged until the end of this quarter.",
        now + 2.5,
    )
    combined = pipeline._get_combined_pending_text()
    assert "announced that the central bank" in combined
    assert pipeline._is_pending_grammatically_complete()
    assert pipeline._should_flush_pending(
        now=now + 2.5, is_currently_speaking=False, post_vad_silence_sec=0.0
    )

    ev = pipeline._flush_pending_to_event()
    assert ev is not None
    assert ev.is_final is True
    assert "announced that the central bank" in ev.en
    assert len(ev.zh) > 6
    print(f"[Merged Full Sentence] EN: {ev.en}\n[Merged Full Sentence] ZH: {ev.zh}")


def test_qwen3_unified_multilingual_asr() -> None:
    pipeline = StreamingSpeechPipeline()
    assert pipeline.use_qwen3_asr is True, "Expected unified Qwen3-ASR-0.6B-int8 model to be active"

    # French dangling-word clause merging
    pipeline.reset_stream(channel_name="France 24国际新闻")
    now = 2000.0
    pipeline._append_clause_to_pending("À Paris aujourd'hui, le président a déclaré que.", now)
    assert not pipeline._is_pending_grammatically_complete()
    pipeline._append_clause_to_pending(
        "Le gouvernement présentera un nouveau plan économique la semaine prochaine.",
        now + 2.0,
    )
    combined_fr = pipeline._get_combined_pending_text()
    assert "déclaré que le gouvernement" in combined_fr
    assert pipeline._is_pending_grammatically_complete()

    # Run actual Qwen3-ASR-0.6B inference with zero language configuration
    with wave.open(str(TEST_WAV), "rb") as wf:
        samples = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    recognized = pipeline._recognize_sensevoice(samples)
    assert len(recognized) > 10
    assert pipeline.last_asr_engine == "Qwen3-ASR-0.6B"
    print(f"[Qwen3-ASR-0.6B Output] ({pipeline._pending_last_asr_ms:.0f}ms): {recognized}")


def test_concurrent_reset_and_long_segment_safety() -> None:
    import threading

    pipeline = StreamingSpeechPipeline()
    with wave.open(str(TEST_WAV), "rb") as wf:
        samples = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0

    # 1. Test 16-second overly long audio array cap in _recognize_sensevoice
    long_samples = np.tile(samples, 4)[: 16000 * 16]
    res = pipeline._recognize_sensevoice(long_samples)
    assert isinstance(res, str) and len(res) > 5

    # 2. Test concurrent feed_samples + reset_stream across threads (previously caused C++ SIGSEGV)
    errors: list[Exception] = []

    def feeder() -> None:
        try:
            for _ in range(6):
                pipeline.feed_samples(samples[:16000])
        except Exception as e:
            errors.append(e)

    def resetter() -> None:
        try:
            for i in range(6):
                pipeline.reset_stream(channel_name=f"Channel-{i}")
        except Exception as e:
            errors.append(e)

    t1 = threading.Thread(target=feeder)
    t2 = threading.Thread(target=resetter)
    t1.start()
    t2.start()
    t1.join(timeout=15)
    t2.join(timeout=15)
    assert not errors, f"Concurrent ASR/VAD access raised errors: {errors}"



