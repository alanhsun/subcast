"""
Stream pass-through proxy (HLS .m3u8 & HTTP-TS) and single-instance FFmpeg AudioSidecarTap for 声幕 SubCast.
Automatically tracks the currently watched channel on the TV, extracts 16kHz mono PCM
audio in the background, runs the local AI subtitle pipeline (Silero-VAD + Qwen3-ASR-0.6B
+ LLM Context Translator), and broadcasts subtitles with configurable delay compensation.
"""

from __future__ import annotations

import asyncio
import queue
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, List, Optional
from urllib.parse import quote, urljoin

from server.ai_pipeline import StreamingSpeechPipeline, SubtitleEvent
from server.playlist_parser import encode_url_token

_URI_ATTR_RE = re.compile(r'URI="([^"]+)"')


def rewrite_m3u8_manifest(
    manifest_text: str,
    final_upstream_url: str,
    base_proxy_url: str,
    channel_id: str,
    channel_name: str,
    proxy_segments: bool = False,
) -> str:
    base = base_proxy_url.rstrip("/")
    encoded_name = quote(channel_name)
    out_lines: List[str] = []

    def _wrap_proxy(target_url: str, is_playlist: bool) -> str:
        tok = encode_url_token(target_url)
        ext = "index.m3u8" if is_playlist else "stream.ts"
        return f"{base}/live/{channel_id}/{ext}?u={tok}&name={encoded_name}"

    lines = manifest_text.splitlines()
    expect_variant_playlist = False

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            out_lines.append("")
            continue

        if line.startswith("#"):
            if line.upper().startswith("#EXT-X-STREAM-INF"):
                expect_variant_playlist = True
            if 'URI="' in line:
                is_media_tag = line.upper().startswith("#EXT-X-MEDIA")

                def _replace_uri(match: re.Match) -> str:
                    orig_uri = match.group(1)
                    abs_uri = urljoin(final_upstream_url, orig_uri)
                    if is_media_tag or abs_uri.split("?")[0].lower().endswith(".m3u8"):
                        return f'URI="{_wrap_proxy(abs_uri, True)}"'
                    return f'URI="{abs_uri}"'

                line = _URI_ATTR_RE.sub(_replace_uri, line)
            out_lines.append(line)
            continue

        abs_url = urljoin(final_upstream_url, line)
        lower_path = abs_url.split("?")[0].lower()
        is_sub_m3u8 = expect_variant_playlist or lower_path.endswith(".m3u8")
        expect_variant_playlist = False

        if is_sub_m3u8:
            out_lines.append(_wrap_proxy(abs_url, True))
        elif proxy_segments:
            out_lines.append(_wrap_proxy(abs_url, False))
        else:
            out_lines.append(abs_url)

    return "\n".join(out_lines) + "\n"


@dataclass
class SidecarStatus:
    active: bool
    channel_id: str
    channel_name: str
    upstream_url: str
    manual_pinned: bool
    subtitle_delay_ms: int
    display_mode: str  # "bilingual" | "zh_only"
    last_touch_ago_sec: float
    total_subtitles_emitted: int
    asr_engine: str = "Silero-VAD + Qwen3-ASR-0.6B"
    translation_engine: str = "CTranslate2"
    llm_ready: bool = False
    llm_error: str = ""
    sentence_mode: str = "full_sentence"
    auto_sync: bool = True
    auto_stream_lead_ms: int = 3500


def extract_hls_segment_duration(manifest_text: str) -> Optional[float]:
    """Extracts average segment duration (seconds) from #EXTINF or #EXT-X-TARGETDURATION in an M3U8."""
    extinf_vals: List[float] = []
    target_dur: Optional[float] = None
    for raw_line in manifest_text.splitlines():
        line = raw_line.strip()
        if line.upper().startswith("#EXTINF:"):
            val_str = line.split(":", 1)[1].split(",", 1)[0].strip()
            try:
                v = float(val_str)
                if 0.4 <= v <= 15.0:
                    extinf_vals.append(v)
            except ValueError:
                pass
        elif line.upper().startswith("#EXT-X-TARGETDURATION:"):
            val_str = line.split(":", 1)[1].strip()
            try:
                target_dur = float(val_str)
            except ValueError:
                pass
    if extinf_vals:
        recent = extinf_vals[-4:]
        return sum(recent) / len(recent)
    return target_dur


class AudioSidecarManager:
    """
    Manages the single active FFmpeg audio extraction process and feeds 16kHz PCM frames
    into StreamingSpeechPipeline on a worker thread, then schedules delayed WebSocket broadcasts.
    """

    def __init__(
        self,
        pipeline: StreamingSpeechPipeline,
        broadcast_cb: Callable[[dict], Awaitable[None]],
        idle_timeout_sec: float = 25.0,
    ) -> None:
        self.pipeline = pipeline
        self.broadcast_cb = broadcast_cb
        self.idle_timeout_sec = idle_timeout_sec

        self.active_channel_id: str = ""
        self.active_channel_name: str = ""
        self.active_upstream_url: str = ""
        self.last_touch_time: float = 0.0
        self.manual_pinned: bool = False

        self.subtitle_delay_ms: int = 800
        self.display_mode: str = "bilingual"
        self.font_scale: float = 1.0
        self.sentence_mode: str = "full_sentence"
        self.auto_sync: bool = True
        self.auto_stream_lead_ms: int = 3500

        self.total_subtitles_emitted: int = 0
        self.recent_subtitles: List[dict] = []
        self._last_subtitle_payload: Optional[dict] = None
        self._last_subtitle_content_ts: float = 0.0
        self._last_broadcast_ts: float = 0.0

        self._ffmpeg_proc: Optional[subprocess.Popen] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._translate_thread: Optional[threading.Thread] = None
        self._translate_queue: queue.Queue[Optional[SubtitleEvent]] = queue.Queue(maxsize=32)
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._watchdog_task: Optional[asyncio.Task] = None

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        if self._watchdog_task is None or self._watchdog_task.done():
            self._watchdog_task = loop.create_task(self._idle_watchdog_loop())

    def touch_channel(
        self,
        channel_id: str,
        channel_name: str,
        upstream_url: str,
        manual_pin: Optional[bool] = None,
    ) -> None:
        now = time.time()
        with self._lock:
            self.last_touch_time = now
            if manual_pin is not None:
                self.manual_pinned = manual_pin

            is_same = (
                self._worker_thread is not None
                and self._worker_thread.is_alive()
                and self.active_upstream_url == upstream_url
            )
            if is_same:
                self.active_channel_name = channel_name or self.active_channel_name
                return

            self._stop_worker_locked()
            self.active_channel_id = channel_id
            self.active_channel_name = channel_name or "Live Channel"
            self.active_upstream_url = upstream_url
            self._last_subtitle_payload = None
            self._stop_event.clear()
            while not self._translate_queue.empty():
                try:
                    self._translate_queue.get_nowait()
                except queue.Empty:
                    break
            self._translate_thread = threading.Thread(
                target=self._translation_worker,
                daemon=True,
                name=f"LLMTrans-{channel_id}",
            )
            self._translate_thread.start()
            self._worker_thread = threading.Thread(
                target=self._audio_tap_worker,
                args=(channel_id, self.active_channel_name, upstream_url),
                daemon=True,
                name=f"AudioTap-{channel_id}",
            )
            self._worker_thread.start()

    def stop_current(self) -> None:
        with self._lock:
            self.manual_pinned = False
            self._stop_worker_locked()
            self.active_channel_id = ""
            self.active_channel_name = ""
            self.active_upstream_url = ""

    def _stop_worker_locked(self) -> None:
        self._stop_event.set()
        self._last_subtitle_payload = None
        try:
            self._translate_queue.put_nowait(None)
        except queue.Full:
            pass
        proc = self._ffmpeg_proc
        self._ffmpeg_proc = None
        if proc is not None:
            try:
                proc.kill()
            except Exception:
                pass

    def update_hls_manifest_metrics(self, manifest_text: str) -> None:
        seg_dur = extract_hls_segment_duration(manifest_text)
        if seg_dur is not None:
            # TV players (ExoPlayer/IJK) typically buffer ~2.0 segments behind FFmpeg's live_start_index=-1 edge
            estimated_lead = int(round(seg_dur * 1900.0))
            self.auto_stream_lead_ms = max(2400, min(6800, estimated_lead))

    def get_status(self) -> SidecarStatus:
        with self._lock:
            is_alive = self._worker_thread is not None and self._worker_thread.is_alive()
            ago = round(time.time() - self.last_touch_time, 1) if self.last_touch_time > 0 else -1.0
            if getattr(self.pipeline, "use_qwen3_asr", False):
                asr_name = "Qwen3-ASR-0.6B"
            elif self.pipeline.use_sensevoice:
                asr_name = "SenseVoice-Small"
            else:
                asr_name = "Zipformer-En"
            llm_tr = self.pipeline.llm_translator
            tr_name = llm_tr.config.model if llm_tr.is_llm_ready else "CTranslate2"
            return SidecarStatus(
                active=is_alive,
                channel_id=self.active_channel_id if is_alive else "",
                channel_name=self.active_channel_name if is_alive else "",
                upstream_url=self.active_upstream_url if is_alive else "",
                manual_pinned=self.manual_pinned and is_alive,
                subtitle_delay_ms=self.subtitle_delay_ms,
                display_mode=self.display_mode,
                last_touch_ago_sec=ago,
                total_subtitles_emitted=self.total_subtitles_emitted,
                asr_engine=asr_name,
                translation_engine=tr_name,
                llm_ready=llm_tr.is_llm_ready,
                llm_error=llm_tr.last_error,
                sentence_mode=self.pipeline.sentence_mode,
                auto_sync=self.auto_sync,
                auto_stream_lead_ms=self.auto_stream_lead_ms,
            )

    async def _idle_watchdog_loop(self) -> None:
        while True:
            await asyncio.sleep(2.5)
            keepalive_payload: Optional[dict] = None
            now = time.time()
            with self._lock:
                is_alive = self._worker_thread is not None and self._worker_thread.is_alive()
                if is_alive and self._last_subtitle_payload is not None:
                    # Keep the current subtitle alive on the TV screen so it never blanks out
                    # while the next full sentence is being spoken and translated!
                    if (
                        (now - self._last_broadcast_ts) >= 4.0
                        and (now - self._last_subtitle_content_ts) <= 38.0
                    ):
                        keepalive_payload = dict(self._last_subtitle_payload)
                        self._last_broadcast_ts = now

                if not is_alive or self.manual_pinned:
                    pass
                elif (now - self.last_touch_time) > self.idle_timeout_sec:
                    print(
                        f"[Sidecar] Channel '{self.active_channel_name}' idle for "
                        f">{self.idle_timeout_sec}s, stopping FFmpeg audio tap."
                    )
                    self._stop_worker_locked()
                    self.active_channel_id = ""
                    self.active_channel_name = ""
                    self.active_upstream_url = ""

            if keepalive_payload is not None:
                await self.broadcast_cb(keepalive_payload)

    def _translation_worker(self) -> None:
        """Background worker that runs LLM translations sequentially without blocking FFmpeg audio capture."""
        while not self._stop_event.is_set():
            try:
                ev = self._translate_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if ev is None or self._stop_event.is_set():
                break
            try:
                if not ev.zh:
                    ev = self.pipeline.complete_event_translation(ev)
                if ev.zh and not self._stop_event.is_set():
                    self._handle_subtitle_event(ev)
            except Exception as exc:
                print(f"[Sidecar] Translation worker error: {exc}")

    def _audio_tap_worker(self, channel_id: str, channel_name: str, upstream_url: str) -> None:
        ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"
        print(f"[Sidecar] Starting audio tap for [{channel_name}] -> {upstream_url}")
        self.pipeline.reset_stream(channel_name=channel_name)
        self.pipeline.async_translate_mode = True

        self._schedule_broadcast(
            {
                "type": "status",
                "event": "channel_switch",
                "channel": channel_name,
                "en": f"Listening: {channel_name}",
                "zh": f"声幕已接入频道：{channel_name}",
                "is_final": True,
                "display_mode": self.display_mode,
            },
            delay_ms=0,
        )

        is_http = upstream_url.lower().startswith(("http://", "https://"))
        is_hls = ".m3u8" in upstream_url.lower()
        cmd = [ffmpeg_bin, "-hide_banner", "-loglevel", "error"]
        if is_http:
            cmd.extend(
                [
                    "-reconnect",
                    "1",
                    "-reconnect_streamed",
                    "1",
                    "-reconnect_delay_max",
                    "4",
                    "-rw_timeout",
                    "12000000",
                    "-user_agent",
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) LinguaRelayTV/2.0",
                ]
            )
        if is_hls:
            # Start FFmpeg audio tap at the newest live edge segment (-1) so AI transcription
            # naturally runs 3~6s ahead of the TV video player's 2~3 segment buffer.
            cmd.extend(["-live_start_index", "-1"])
        cmd.extend(
            [
                "-i",
                upstream_url,
                "-vn",
                "-sn",
                "-dn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-f",
                "s16le",
                "pipe:1",
            ]
        )

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=6400,
            )
            with self._lock:
                if self._stop_event.is_set():
                    proc.kill()
                    return
                self._ffmpeg_proc = proc

            chunk_size = 3200  # 100ms of 16kHz 16-bit mono PCM
            assert proc.stdout is not None
            while not self._stop_event.is_set():
                pcm_chunk = proc.stdout.read(chunk_size)
                if not pcm_chunk:
                    break
                events = self.pipeline.feed_pcm_s16le(pcm_chunk)
                for ev in events:
                    try:
                        self._translate_queue.put_nowait(ev)
                    except queue.Full:
                        pass

        except Exception as exc:
            print(f"[Sidecar] Audio tap error on [{channel_name}]: {exc}")
        finally:
            with self._lock:
                if self._ffmpeg_proc is not None:
                    try:
                        self._ffmpeg_proc.kill()
                    except Exception:
                        pass
                    self._ffmpeg_proc = None
            print(f"[Sidecar] Audio tap stopped for [{channel_name}].")

    def _handle_subtitle_event(self, ev: SubtitleEvent) -> None:
        self.total_subtitles_emitted += 1
        if self.auto_sync and ev.pipeline_elapsed_ms > 0:
            # Dynamic per-sentence compensation:
            # Target alignment = auto_stream_lead_ms + user fine-tune offset (subtitle_delay_ms - 800)
            # Subtract the actual time already spent in VAD + clause aggregation + ASR + LLM!
            fine_tune_ms = self.subtitle_delay_ms - 800
            target_lead_ms = self.auto_stream_lead_ms + fine_tune_ms
            effective_delay_ms = max(0, int(round(target_lead_ms - ev.pipeline_elapsed_ms)))
        else:
            effective_delay_ms = self.subtitle_delay_ms

        payload = {
            "type": "subtitle",
            "channel": ev.channel,
            "en": ev.en,
            "zh": ev.zh,
            "is_final": ev.is_final,
            "timestamp": ev.timestamp,
            "latency_ms": ev.latency_ms,
            "pipeline_elapsed_ms": ev.pipeline_elapsed_ms,
            "actual_delay_ms": effective_delay_ms,
            "stream_lead_ms": self.auto_stream_lead_ms,
            "auto_sync": self.auto_sync,
            "engine_info": ev.engine_info,
            "display_mode": self.display_mode,
            "font_scale": self.font_scale,
        }
        if ev.is_final:
            self.recent_subtitles.append(payload)
            if len(self.recent_subtitles) > 60:
                self.recent_subtitles = self.recent_subtitles[-60:]
        self._schedule_broadcast(payload, delay_ms=effective_delay_ms)

    def _schedule_broadcast(self, payload: dict, delay_ms: int) -> None:
        if self._loop is None or self._loop.is_closed():
            return

        async def _delayed_send() -> None:
            if delay_ms > 0:
                await asyncio.sleep(delay_ms / 1000.0)
            now = time.time()
            with self._lock:
                self._last_broadcast_ts = now
                if payload.get("type") == "subtitle":
                    self._last_subtitle_payload = payload
                    self._last_subtitle_content_ts = now
            await self.broadcast_cb(payload)

        asyncio.run_coroutine_threadsafe(_delayed_send(), self._loop)
