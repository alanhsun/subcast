"""
Stream pass-through proxy (HLS .m3u8 & HTTP-TS) and single-instance FFmpeg AudioSidecarTap for 声幕 SubCast.
Automatically tracks the currently watched channel on the TV, extracts 16kHz mono PCM
audio in the background, runs the local AI subtitle pipeline (Silero-VAD + Qwen3-ASR-0.6B
+ LLM Context Translator), and broadcasts subtitles with configurable delay compensation.
"""

from __future__ import annotations

import asyncio
import os
import queue
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, List, Optional
from urllib.parse import parse_qs, quote, unquote, urljoin, urlparse

import httpx

from server.ai_pipeline import StreamingSpeechPipeline, SubtitleEvent
from server.playlist_parser import encode_url_token

_URI_ATTR_RE = re.compile(r'URI="([^"]+)"')
_BANDWIDTH_RE = re.compile(r"BANDWIDTH=(\d+)", re.IGNORECASE)
_FFMPEG_EXTRA_ARGS_CACHE: dict[str, List[str]] = {}


def _unwrap_beacon_url(url: str) -> str:
    """Unwraps Amagi/Rakuten /beacon/...?redirect_url=... tracking URLs to direct .ts/.m3u8 URLs."""
    if "redirect_url=" not in url:
        return url
    try:
        parsed = urlparse(url)
        qs = parse_qs(parsed.query)
        redir = qs.get("redirect_url", [""])[0]
        if redir:
            unq = unquote(redir)
            if unq.startswith(("http://", "https://")):
                return unq
    except Exception:
        pass
    return url


def _discover_sibling_master_urls(base_url: str) -> List[str]:
    """
    Given a single-bitrate media M3U8 URL (e.g. .../master_6500000.m3u8, .../01.m3u8,
    .../chunklist_b4500000.m3u8, or .../highw/index.m3u8), returns candidate Master M3U8 URLs
    on the same origin/feed so FFmpeg can tap a low-bitrate or audio-only sibling track.
    """
    try:
        parsed = urlparse(base_url)
        path = parsed.path or ""
    except Exception:
        return []

    candidates: List[str] = []

    def _add(rel: str) -> None:
        u = urljoin(base_url, rel)
        if u != base_url and u not in candidates:
            candidates.append(u)

    if re.search(r"/(?:master|index|playlist|prog_index)_[A-Za-z0-9_-]+\.m3u8$", path, re.IGNORECASE):
        for name in ("master.m3u8", "playlist.m3u8", "index.m3u8"):
            _add(name)
    elif re.search(r"/0[1-6]\.m3u8$", path, re.IGNORECASE):
        for name in ("master.m3u8", "index.m3u8", "playlist.m3u8"):
            _add(name)
    elif re.search(r"/chunklist_b\d+.*\.m3u8$", path, re.IGNORECASE):
        for name in ("playlist.m3u8", "master.m3u8"):
            _add(name)

    if re.search(
        r"/(?:stream\d+|\w*high\w*|hd|\d{3,4}p|b?\d{6,8})/(?:index|playlist|streamPlaylist|prog_index)\.m3u8$",
        path,
        re.IGNORECASE,
    ):
        for name in ("../master.m3u8", "../playlist.m3u8", "../index.m3u8"):
            _add(name)

    return candidates[:3]


def resolve_best_audio_stream_info(
    url: str,
    timeout: float = 5.5,
    stream_proxy: str = "",
) -> tuple[str, Optional[float]]:
    """
    Resolves a Master M3U8 (or single-bitrate variant with a sibling Master M3U8) into its
    single best audio sub-playlist URL and extracts its average segment duration (seconds) so FFmpeg does not:
    1) Download all 4~8 video resolution variants simultaneously (causing 15-25s timeouts),
    2) Pull a 4-6 Mbps 1080p stream across slow cross-border links when a 64-480 kbps sibling exists,
    3) Miss demuxed audio tracks (#EXT-X-MEDIA:TYPE=AUDIO,URI="..."), or
    4) Abort on dead subtitle sub-playlists (#EXT-X-MEDIA:TYPE=SUBTITLES -> 404).
    """
    clean_url = _unwrap_beacon_url(url)
    if ".m3u8" not in clean_url.lower():
        return clean_url, None
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) LinguaRelayTV/2.4"}
    client_kwargs: dict = {"follow_redirects=True": True}
    client_opts: dict = {"follow_redirects": True, "timeout": timeout, "verify": False}
    if stream_proxy:
        client_opts["proxy"] = stream_proxy
    del client_kwargs
    try:
        with httpx.Client(**client_opts) as client:
            resp = client.get(clean_url, headers=headers)
            if resp.status_code >= 400:
                return clean_url, None
            base_url = str(resp.url)
            text = resp.text
            if "#EXT-X-STREAM-INF" not in text and "#EXT-X-MEDIA" not in text:
                for sib_url in _discover_sibling_master_urls(base_url):
                    try:
                        r_sib = client.get(sib_url, headers=headers, timeout=2.5)
                        if r_sib.status_code < 400 and (
                            "#EXT-X-STREAM-INF" in r_sib.text or "#EXT-X-MEDIA" in r_sib.text
                        ):
                            base_url = str(r_sib.url)
                            text = r_sib.text
                            break
                    except Exception:
                        pass
                if "#EXT-X-STREAM-INF" not in text and "#EXT-X-MEDIA" not in text:
                    return base_url, extract_hls_segment_duration(text)

            lines = [line.strip() for line in text.splitlines() if line.strip()]

            # 1. Check for dedicated #EXT-X-MEDIA:TYPE=AUDIO playlist (e.g. NHK World, NTV News24, Weathernews, TalkTV)
            audio_candidates: List[tuple[int, str]] = []
            for line in lines:
                if line.upper().startswith("#EXT-X-MEDIA") and "TYPE=AUDIO" in line.upper():
                    m = _URI_ATTR_RE.search(line)
                    if m:
                        a_url = _unwrap_beacon_url(urljoin(base_url, m.group(1)))
                        prio = 0 if "DEFAULT=YES" in line.upper() else 1
                        audio_candidates.append((prio, a_url))
            audio_candidates.sort(key=lambda x: x[0])
            for _, a_url in audio_candidates:
                try:
                    r_aud = client.get(a_url, headers=headers)
                    if r_aud.status_code < 400 and "#EXTINF" in r_aud.text:
                        return str(r_aud.url), extract_hls_segment_duration(r_aud.text)
                except Exception:
                    pass

            # 2. Otherwise pick the lowest-bandwidth working #EXT-X-STREAM-INF variant for fast audio extraction
            variants: List[tuple[int, str]] = []
            for i, line in enumerate(lines):
                if line.upper().startswith("#EXT-X-STREAM-INF"):
                    bw_m = _BANDWIDTH_RE.search(line)
                    bw = int(bw_m.group(1)) if bw_m else 99999999
                    for j in range(i + 1, len(lines)):
                        if not lines[j].startswith("#"):
                            v_url = _unwrap_beacon_url(urljoin(base_url, lines[j]))
                            variants.append((bw, v_url))
                            break
            if variants:
                variants.sort(key=lambda x: x[0])
                for _, v_url in variants[:3]:
                    try:
                        r_var = client.get(v_url, headers=headers)
                        if r_var.status_code < 400 and "#EXTM3U" in r_var.text.upper():
                            return str(r_var.url), extract_hls_segment_duration(r_var.text)
                    except Exception:
                        pass
                return variants[0][1], None
    except Exception:
        pass
    return clean_url, None


def resolve_best_audio_stream_url(url: str, timeout: float = 5.5, stream_proxy: str = "") -> str:
    return resolve_best_audio_stream_info(url, timeout=timeout, stream_proxy=stream_proxy)[0]


def _get_ffmpeg_extra_args(ffmpeg_bin: str) -> List[str]:
    """Detects FFmpeg version support for HLS extension bypass flags (FFmpeg 5.x/6.x/7.x)."""
    if ffmpeg_bin in _FFMPEG_EXTRA_ARGS_CACHE:
        return list(_FFMPEG_EXTRA_ARGS_CACHE[ffmpeg_bin])
    extra: List[str] = []
    try:
        res = subprocess.run(
            [ffmpeg_bin, "-hide_banner", "-h", "demuxer=hls"],
            capture_output=True,
            text=True,
            timeout=3.0,
        )
        help_text = (res.stdout or "") + (res.stderr or "")
        if "extension_picky" in help_text:
            extra.extend(["-extension_picky", "0"])
        if "allowed_extensions" in help_text:
            extra.extend(["-allowed_extensions", "ALL"])
        if "allowed_segment_extensions" in help_text:
            extra.extend(["-allowed_segment_extensions", "ALL"])
    except Exception:
        pass
    _FFMPEG_EXTRA_ARGS_CACHE[ffmpeg_bin] = extra
    return list(extra)


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
                upper_line = line.upper()
                is_media_tag = upper_line.startswith("#EXT-X-MEDIA")
                is_subtitle_media = is_media_tag and "TYPE=SUBTITLES" in upper_line

                def _replace_uri(match: re.Match) -> str:
                    orig_uri = match.group(1)
                    abs_uri = _unwrap_beacon_url(urljoin(final_upstream_url, orig_uri))
                    if is_subtitle_media:
                        return f'URI="{abs_uri}"'
                    if is_media_tag or abs_uri.split("?")[0].lower().endswith(".m3u8"):
                        return f'URI="{_wrap_proxy(abs_uri, True)}"'
                    return f'URI="{abs_uri}"'

                line = _URI_ATTR_RE.sub(_replace_uri, line)
            out_lines.append(line)
            continue

        abs_url = _unwrap_beacon_url(urljoin(final_upstream_url, line))
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


def _terminate_ffmpeg_proc(proc: Optional[subprocess.Popen]) -> None:
    """Safely kill an FFmpeg subprocess, close its stdout pipe, and reap it so no zombie remains."""
    if proc is None:
        return
    try:
        proc.kill()
    except Exception:
        pass
    try:
        if proc.stdout is not None:
            proc.stdout.close()
    except Exception:
        pass
    try:
        proc.wait(timeout=1.5)
    except Exception:
        pass


def _is_master_like_m3u8(url: str) -> bool:
    lower = url.split("?")[0].lower()
    if not lower.endswith(".m3u8"):
        return False
    if re.search(r"(?:_\d{4,}|highw|1080p?|720p?|chunklist_b\d+|/0[1-6]\.m3u8$)", lower):
        return False
    return any(lower.endswith(s) for s in ("/master.m3u8", "/playlist.m3u8", "/index.m3u8"))


def _rank_audio_candidates(primary_url: str, candidates: List[str], slow_urls: set[str]) -> List[str]:
    """
    Orders candidate stream URLs for FFmpeg audio extraction:
    1. Keeps the requested `primary_url` first unless it was previously flagged as too slow (<0.68x).
    2. Prioritizes non-slow URLs and adaptive Master M3U8s (which offer low-bitrate/audio-only variants)
       ahead of single-bitrate HD streams.
    """
    dedup: List[str] = []
    for u in [primary_url] + list(candidates or []):
        if u and u not in dedup:
            dedup.append(u)
    if len(dedup) <= 1:
        return dedup

    def _sort_key(idx_and_url: tuple[int, str]) -> tuple[int, int, int]:
        idx, u = idx_and_url
        is_slow = 1 if u in slow_urls else 0
        if idx == 0 and not is_slow:
            return (0, 0, 0)
        not_master = 0 if _is_master_like_m3u8(u) else 1
        return (is_slow, 1 + not_master, idx)

    ranked = [u for _, u in sorted(enumerate(dedup), key=_sort_key)]
    return ranked


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
        self.stream_proxy: str = os.environ.get("LINGUARELAY_STREAM_PROXY", "").strip()

        self.active_channel_id: str = ""
        self.active_channel_name: str = ""
        self.active_upstream_url: str = ""
        self.active_candidate_urls: List[str] = []
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
        self._session_epoch: int = 0
        self._slow_urls: set[str] = set()

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
        candidate_urls: Optional[List[str]] = None,
    ) -> None:
        now = time.time()
        old_worker: Optional[threading.Thread] = None
        with self._lock:
            self.last_touch_time = now
            if manual_pin is not None:
                self.manual_pinned = manual_pin

            is_same = (
                self._worker_thread is not None
                and self._worker_thread.is_alive()
                and (
                    self.active_upstream_url == upstream_url
                    or (
                        bool(channel_id)
                        and channel_id != "manual"
                        and self.active_channel_id == channel_id
                    )
                )
            )
            if is_same:
                self.active_channel_name = channel_name or self.active_channel_name
                return

            dedup_candidates = _rank_audio_candidates(
                upstream_url, list(candidate_urls or []), self._slow_urls
            )

            old_worker = self._worker_thread
            self._stop_worker_locked()
            session_epoch = self._session_epoch
            self.active_channel_id = channel_id
            self.active_channel_name = channel_name or "Live Channel"
            self.active_upstream_url = dedup_candidates[0] if dedup_candidates else upstream_url
            self.active_candidate_urls = dedup_candidates
            self._last_subtitle_payload = None

            # Allocate a brand-new Event and Queue per channel session so the stopping
            # worker threads never observe a cleared _stop_event or drained poison pill.
            session_stop = threading.Event()
            session_queue: queue.Queue[Optional[SubtitleEvent]] = queue.Queue(maxsize=32)
            self._stop_event = session_stop
            self._translate_queue = session_queue

            # Clear any previous channel's subtitle on the TV screen immediately
            self._schedule_broadcast({"type": "clear"}, delay_ms=0, session_epoch=None)

            self._translate_thread = threading.Thread(
                target=self._translation_worker,
                args=(session_stop, session_queue, session_epoch),
                daemon=True,
                name=f"LLMTrans-{channel_id}",
            )
            self._translate_thread.start()
            self._worker_thread = threading.Thread(
                target=self._audio_tap_worker,
                args=(
                    channel_id,
                    self.active_channel_name,
                    upstream_url,
                    dedup_candidates,
                    session_stop,
                    session_queue,
                    old_worker,
                    session_epoch,
                ),
                daemon=True,
                name=f"AudioTap-{channel_id}",
            )
            self._worker_thread.start()

    def stop_current(self) -> bool:
        with self._lock:
            was_active = bool(self.active_channel_id) or (
                self._worker_thread is not None and self._worker_thread.is_alive()
            )
            self.manual_pinned = False
            self._stop_worker_locked()
            self.active_channel_id = ""
            self.active_channel_name = ""
            self.active_upstream_url = ""
            self.active_candidate_urls = []
            return was_active

    def _stop_worker_locked(self) -> None:
        self._session_epoch += 1
        self._stop_event.set()
        self._last_subtitle_payload = None
        self._last_subtitle_content_ts = 0.0
        try:
            self._translate_queue.put_nowait(None)
        except queue.Full:
            pass
        proc = self._ffmpeg_proc
        self._ffmpeg_proc = None
        _terminate_ffmpeg_proc(proc)

    def update_hls_manifest_metrics_from_seg_dur(self, seg_dur: float) -> None:
        """
        Calculates the stream lead (ms) between our paced FFmpeg audio tap and the TV video player.
        - For seg_dur < 6.0s: FFmpeg starts at -live_start_index -1 (with a 3.5s initial lookahead burst,
          then paced at 1.0x real-time), while TV players (ExoPlayer/IJK) buffer at ~ -2.5..-3.
        - For seg_dur >= 6.0s (e.g. 10s segments on Al Jazeera): FFmpeg starts at -live_start_index -2
          (with a 3.5s initial burst, then 1.0x real-time pacing), keeping lead around ~12.5s instead of 25s.
        """
        if seg_dur >= 6.0:
            estimated_lead = int(round(seg_dur * 950.0 + 3500.0))
        else:
            estimated_lead = int(round(seg_dur * 1350.0 + min(3500.0, seg_dur * 800.0)))
        self.auto_stream_lead_ms = max(2800, min(18000, estimated_lead))

    def update_hls_manifest_metrics(self, manifest_text: str) -> None:
        seg_dur = extract_hls_segment_duration(manifest_text)
        if seg_dur is not None:
            self.update_hls_manifest_metrics_from_seg_dur(seg_dur)

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
            keepalive_epoch: int = -1
            send_clear = False
            now = time.time()
            with self._lock:
                is_alive = self._worker_thread is not None and self._worker_thread.is_alive()
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
                    self.active_candidate_urls = []
                    is_alive = False
                    send_clear = True

                if is_alive and self._last_subtitle_payload is not None:
                    # Keep the current subtitle alive on the TV screen so it never blanks out
                    # while the next full sentence is being spoken and translated!
                    if (
                        (now - self._last_broadcast_ts) >= 4.0
                        and (now - self._last_subtitle_content_ts) <= 38.0
                    ):
                        keepalive_payload = dict(self._last_subtitle_payload)
                        keepalive_epoch = self._session_epoch
                        self._last_broadcast_ts = now

            if send_clear:
                await self.broadcast_cb({"type": "clear"})
            elif keepalive_payload is not None:
                with self._lock:
                    still_valid = self._session_epoch == keepalive_epoch
                if still_valid:
                    await self.broadcast_cb(keepalive_payload)

    def _translation_worker(
        self,
        stop_event: Optional[threading.Event] = None,
        translate_queue: Optional[queue.Queue[Optional[SubtitleEvent]]] = None,
        session_epoch: Optional[int] = None,
    ) -> None:
        """Background worker that runs LLM translations sequentially without blocking FFmpeg audio capture."""
        ev_stop = stop_event or self._stop_event
        ev_queue = translate_queue or self._translate_queue
        while not ev_stop.is_set():
            try:
                ev = ev_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if ev is None or ev_stop.is_set():
                break
            try:
                if not ev.zh:
                    ev = self.pipeline.complete_event_translation(ev)
                if ev.zh and not ev_stop.is_set():
                    self._handle_subtitle_event(ev, session_epoch=session_epoch)
            except Exception as exc:
                print(f"[Sidecar] Translation worker error: {exc}")

    def _audio_tap_worker(
        self,
        channel_id: str,
        channel_name: str,
        upstream_url: str,
        candidate_urls: Optional[List[str]] = None,
        stop_event: Optional[threading.Event] = None,
        translate_queue: Optional[queue.Queue[Optional[SubtitleEvent]]] = None,
        prev_worker: Optional[threading.Thread] = None,
        session_epoch: Optional[int] = None,
    ) -> None:
        ev_stop = stop_event or self._stop_event
        ev_queue = translate_queue or self._translate_queue
        if prev_worker is not None and prev_worker.is_alive() and prev_worker is not threading.current_thread():
            prev_worker.join(timeout=1.5)
        if ev_stop.is_set():
            return

        ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"
        extra_hls_args = _get_ffmpeg_extra_args(ffmpeg_bin)
        candidates = [u for u in (candidate_urls or [upstream_url]) if u]
        if not candidates:
            candidates = [upstream_url]

        print(
            f"[Sidecar] Starting audio tap for [{channel_name}] "
            f"(candidates={len(candidates)}) -> {candidates[0]}"
        )
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
            session_epoch=session_epoch,
        )

        cand_idx = 0
        try:
            while not ev_stop.is_set():
                raw_url = candidates[cand_idx % len(candidates)]
                cand_idx += 1
                is_http = raw_url.lower().startswith(("http://", "https://"))
                if is_http:
                    target_url, detected_seg_dur = resolve_best_audio_stream_info(
                        raw_url, stream_proxy=self.stream_proxy
                    )
                    if detected_seg_dur is not None:
                        self.update_hls_manifest_metrics_from_seg_dur(detected_seg_dur)
                else:
                    target_url, detected_seg_dur = raw_url, None
                is_hls = ".m3u8" in target_url.lower() or ".m3u8" in raw_url.lower()

                with self._lock:
                    if ev_stop.is_set():
                        return
                    self.active_upstream_url = target_url

                cmd = [
                    ffmpeg_bin,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-probesize",
                    "1000000",
                    "-analyzeduration",
                    "2000000",
                ]
                if is_http:
                    if self.stream_proxy:
                        cmd.extend(["-http_proxy", self.stream_proxy])
                    cmd.extend(
                        [
                            "-reconnect",
                            "1",
                            "-reconnect_streamed",
                            "1",
                            "-reconnect_delay_max",
                            "4",
                            "-rw_timeout",
                            "8000000",
                            "-user_agent",
                            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) LinguaRelayTV/2.4",
                        ]
                    )
                if is_hls:
                    cmd.extend(extra_hls_args)
                    # For long-segment HLS streams (>=6s, e.g. Al Jazeera 10s segments), start at -2
                    # instead of -1 so FFmpeg is ~10-12s ahead of the TV player rather than 20-25s ahead.
                    start_idx = "-2" if (detected_seg_dur and detected_seg_dur >= 6.0) else "-1"
                    cmd.extend(["-live_start_index", start_idx])
                cmd.extend(
                    [
                        "-i",
                        target_url,
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

                proc: Optional[subprocess.Popen] = None
                try:
                    proc = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL,
                        bufsize=6400,
                    )
                    with self._lock:
                        if ev_stop.is_set():
                            _terminate_ffmpeg_proc(proc)
                            return
                        self._ffmpeg_proc = proc

                    chunk_size = 3200  # 100ms of 16kHz 16-bit mono PCM (32000 bytes/sec)
                    lookahead_sec = 3.5  # Allow 3.5s initial burst to cover AI pipeline latency, then pace 1.0x
                    audio_fed_sec = 0.0
                    tap_start_perf = 0.0
                    proc_spawn_perf = time.perf_counter()
                    last_chunk_perf = proc_spawn_perf
                    win_t0 = 0.0
                    win_audio0 = 0.0
                    slow_warned = False

                    pcm_q: queue.Queue[Optional[bytes]] = queue.Queue(maxsize=96)
                    stdout_pipe = proc.stdout
                    assert stdout_pipe is not None

                    def _stdout_reader(
                        pipe=stdout_pipe,
                        out_q: queue.Queue[Optional[bytes]] = pcm_q,
                        stop_ev: threading.Event = ev_stop,
                    ) -> None:
                        try:
                            while not stop_ev.is_set():
                                data = pipe.read(chunk_size)
                                if not data:
                                    break
                                while not stop_ev.is_set():
                                    try:
                                        out_q.put(data, timeout=0.5)
                                        break
                                    except queue.Full:
                                        continue
                        except Exception:
                            pass
                        finally:
                            try:
                                out_q.put_nowait(None)
                            except queue.Full:
                                pass

                    threading.Thread(
                        target=_stdout_reader,
                        daemon=True,
                        name=f"FFmpegRead-{channel_id}",
                    ).start()

                    while not ev_stop.is_set():
                        try:
                            pcm_chunk = pcm_q.get(timeout=1.0)
                        except queue.Empty:
                            now_p = time.perf_counter()
                            if tap_start_perf == 0.0 and (now_p - proc_spawn_perf) > 8.0:
                                print(
                                    f"[Sidecar] First-audio timeout (>8.0s) on [{channel_name}] ({target_url}), failing over..."
                                )
                                if len(candidates) > 1:
                                    with self._lock:
                                        self._slow_urls.add(raw_url)
                                break
                            if tap_start_perf > 0.0 and (now_p - last_chunk_perf) > 6.5:
                                print(
                                    f"[Sidecar] Audio stream stalled (>6.5s) on [{channel_name}] ({target_url}), failing over..."
                                )
                                if len(candidates) > 1:
                                    with self._lock:
                                        self._slow_urls.add(raw_url)
                                break
                            continue

                        if pcm_chunk is None:
                            break

                        now_p = time.perf_counter()
                        last_chunk_perf = now_p
                        if tap_start_perf == 0.0:
                            tap_start_perf = now_p
                            win_t0 = now_p
                            win_audio0 = 0.0
                        audio_fed_sec += len(pcm_chunk) / 32000.0

                        # Real-time cross-border throughput watchdog:
                        # If an HD stream drips at e.g. 0.07x real-time (35 KB/s), FFmpeg's rw_timeout never fires,
                        # causing 60-90s latency per sentence. Detect <0.68x over a 9s window and fail over!
                        if is_http and win_t0 > 0.0 and (now_p - win_t0) >= 9.0:
                            speed_ratio = (audio_fed_sec - win_audio0) / max(0.1, now_p - win_t0)
                            win_t0 = now_p
                            win_audio0 = audio_fed_sec
                            if speed_ratio < 0.68:
                                if len(candidates) > 1:
                                    print(
                                        f"[Sidecar] Upstream too slow ({speed_ratio:.2f}x < 0.68x real-time) on "
                                        f"[{channel_name}] ({target_url}), failing over to faster candidate..."
                                    )
                                    with self._lock:
                                        self._slow_urls.add(raw_url)
                                    break
                                elif not slow_warned:
                                    slow_warned = True
                                    print(
                                        f"[Sidecar] Warning: single upstream for [{channel_name}] is slow ({speed_ratio:.2f}x real-time)."
                                    )

                        events = self.pipeline.feed_pcm_s16le(pcm_chunk)
                        for ev in events:
                            if ev_stop.is_set():
                                break
                            try:
                                ev_queue.put_nowait(ev)
                            except queue.Full:
                                pass

                        # Real-time 1.0x pacing after initial 3.5s lookahead burst so 10s .ts segments
                        # are never burst-decoded all at once 8-10s ahead of their natural speech time
                        if is_http and audio_fed_sec > lookahead_sec:
                            target_elapsed = audio_fed_sec - lookahead_sec
                            actual_elapsed = time.perf_counter() - tap_start_perf
                            sleep_sec = target_elapsed - actual_elapsed
                            if 0.01 < sleep_sec <= 1.5:
                                if ev_stop.wait(sleep_sec):
                                    break
                            elif sleep_sec < -4.0:
                                tap_start_perf = time.perf_counter() - target_elapsed
                except Exception as inner_exc:
                    print(f"[Sidecar] Audio tap attempt error on [{channel_name}] ({target_url}): {inner_exc}")
                finally:
                    with self._lock:
                        if self._ffmpeg_proc is proc:
                            self._ffmpeg_proc = None
                    _terminate_ffmpeg_proc(proc)

                # If not global-stopped and this is a network stream, fail over / reconnect
                if not ev_stop.is_set() and is_http:
                    if ev_stop.wait(1.0):
                        break
                    print(
                        f"[Sidecar] Stream ended or stalled for [{channel_name}], "
                        f"switching/reconnecting (next candidate #{(cand_idx % len(candidates)) + 1})..."
                    )
                else:
                    break

        except Exception as exc:
            print(f"[Sidecar] Audio tap error on [{channel_name}]: {exc}")
        finally:
            print(f"[Sidecar] Audio tap stopped for [{channel_name}].")

    def _handle_subtitle_event(
        self, ev: SubtitleEvent, session_epoch: Optional[int] = None
    ) -> None:
        with self._lock:
            if session_epoch is not None and session_epoch != self._session_epoch:
                return
            self.total_subtitles_emitted += 1
            auto_sync = self.auto_sync
            sub_delay = self.subtitle_delay_ms
            lead_ms = self.auto_stream_lead_ms
            disp_mode = self.display_mode
            f_scale = self.font_scale

        if auto_sync and ev.pipeline_elapsed_ms > 0:
            # Dynamic per-sentence compensation:
            # Target alignment = auto_stream_lead_ms + clamped user fine-tune offset (subtitle_delay_ms - 800)
            # Subtract the actual time already spent in VAD + clause aggregation + ASR + LLM!
            fine_tune_ms = max(-3500, min(3500, sub_delay - 800))
            target_lead_ms = lead_ms + fine_tune_ms
            effective_delay_ms = max(0, int(round(target_lead_ms - ev.pipeline_elapsed_ms)))
        else:
            effective_delay_ms = sub_delay

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
            "stream_lead_ms": lead_ms,
            "auto_sync": auto_sync,
            "engine_info": ev.engine_info,
            "display_mode": disp_mode,
            "font_scale": f_scale,
        }
        with self._lock:
            if session_epoch is not None and session_epoch != self._session_epoch:
                return
            if ev.is_final:
                self.recent_subtitles.append(payload)
                if len(self.recent_subtitles) > 60:
                    self.recent_subtitles = self.recent_subtitles[-60:]
        self._schedule_broadcast(
            payload, delay_ms=effective_delay_ms, session_epoch=session_epoch
        )

    def _schedule_broadcast(
        self,
        payload: dict,
        delay_ms: int,
        session_epoch: Optional[int] = None,
    ) -> None:
        if self._loop is None or self._loop.is_closed():
            return

        async def _delayed_send() -> None:
            if session_epoch is not None:
                with self._lock:
                    if session_epoch != self._session_epoch:
                        return
            if delay_ms > 0:
                await asyncio.sleep(delay_ms / 1000.0)
            now = time.time()
            with self._lock:
                if session_epoch is not None and session_epoch != self._session_epoch:
                    return
                self._last_broadcast_ts = now
                if payload.get("type") == "subtitle":
                    self._last_subtitle_payload = payload
                    self._last_subtitle_content_ts = now
            await self.broadcast_cb(payload)

        asyncio.run_coroutine_threadsafe(_delayed_send(), self._loop)
