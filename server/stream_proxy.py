"""
Stream pass-through proxy (HLS .m3u8 & HTTP-TS) and single-instance FFmpeg AudioSidecarTap for 声幕 SubCast.
Automatically tracks the currently watched channel on the TV, extracts 16kHz mono PCM
audio in the background, runs the local AI subtitle pipeline (Silero-VAD + Qwen3-ASR-0.6B
+ LLM Context Translator), and broadcasts subtitles with configurable delay compensation.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import math
import os
import posixpath
import queue
import re
import shutil
import subprocess
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Awaitable, Callable, List, Optional
from urllib.parse import parse_qs, quote, unquote, urljoin, urlparse

import httpx

from server.ai_pipeline import StreamingSpeechPipeline, SubtitleEvent
from server.playlist_parser import encode_url_token, sanitize_client_id

_URI_ATTR_RE = re.compile(r'URI="([^"]+)"')
_BANDWIDTH_RE = re.compile(r"BANDWIDTH=(\d+)", re.IGNORECASE)
_HLS_OPENING_RE = re.compile(r"Opening\s+['\"]([^'\"]+)['\"]\s+for\s+reading", re.IGNORECASE)
_SAFE_SEG_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
_VALID_MEDIA_EXTS = {".ts", ".m4s", ".mp4", ".aac", ".cmfa", ".cmfv", ".m4a", ".mp3"}
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


def _parse_iso_utc_epoch(pdt_raw: str) -> Optional[float]:
    """Parses ISO-8601 #EXT-X-PROGRAM-DATE-TIME into UTC Unix epoch seconds."""
    s = pdt_raw.strip()
    if not s:
        return None
    if s.endswith(("Z", "z")):
        s = s[:-1] + "+00:00"
    elif re.search(r"[+-]\d{4}$", s):
        s = s[:-2] + ":" + s[-2:]
    try:
        dt = _dt.datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=_dt.timezone.utc)
        return dt.timestamp()
    except Exception:
        return None


def _norm_seg_path(url: str) -> str:
    try:
        return urlparse(_unwrap_beacon_url(url)).path or url
    except Exception:
        return url


def _safe_seg_filename(abs_url: str, default_name: str = "seg.ts") -> str:
    """Preserves media segment extension (.ts / .m4s / .aac / .mp4) for strict HLS demuxers."""
    try:
        raw_base = posixpath.basename(urlparse(abs_url).path or "")
    except Exception:
        raw_base = ""
    cleaned = _SAFE_SEG_NAME_RE.sub("_", raw_base).strip("._")
    if not cleaned:
        return default_name
    _, ext = posixpath.splitext(cleaned.lower())
    if ext not in _VALID_MEDIA_EXTS:
        cleaned = f"{cleaned[:48]}.ts"
    elif len(cleaned) > 64:
        stem, orig_ext = posixpath.splitext(cleaned)
        cleaned = f"{stem[:56]}{orig_ext}"
    return cleaned


@dataclass
class HlsSegmentMeta:
    seq: int
    url: str
    norm_path: str
    start_pts: float
    duration: float


class HlsTimelineClock:
    """
    Unified per-channel broadcast PTS clock across Master/Variant HLS playlists:
    - Uses exact UTC epoch when `#EXT-X-PROGRAM-DATE-TIME` is present (0ms cross-variant drift).
    - Falls back to `#EXT-X-MEDIA-SEQUENCE` + `#EXTINF` continuous timeline anchored to the upstream
      live edge and shared across ABR variants with aligned sequence numbers.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.uses_pdt: bool = False
        self._url_to_seg: OrderedDict[str, HlsSegmentMeta] = OrderedDict()
        self._path_to_seg: OrderedDict[str, HlsSegmentMeta] = OrderedDict()
        self._init_paths: set[str] = set()
        self._shared_seq_pts: dict[int, float] = {}
        self._shared_avg_dur: float = 0.0
        self._variant_seq_pts: dict[str, dict[int, float]] = {}

    def is_init_segment(self, seg_url: str) -> bool:
        path = _norm_seg_path(seg_url)
        lower = path.lower()
        if lower.endswith(("init.mp4", "init.m4s", "header.mp4", ".key")):
            return True
        with self._lock:
            return path in self._init_paths or seg_url in self._init_paths

    def lookup_segment(self, seg_url: str) -> Optional[HlsSegmentMeta]:
        clean = _unwrap_beacon_url(seg_url)
        path = _norm_seg_path(clean)
        with self._lock:
            return self._url_to_seg.get(clean) or self._path_to_seg.get(path)

    def lookup_segment_pts(self, seg_url: str) -> Optional[float]:
        meta = self.lookup_segment(seg_url)
        return meta.start_pts if meta is not None else None

    @staticmethod
    def expected_ffmpeg_start_pts(segs: List[HlsSegmentMeta], live_start_index: int) -> Optional[float]:
        if not segs:
            return None
        idx = max(0, min(len(segs) - 1, len(segs) + live_start_index))
        return segs[idx].start_pts

    def _assign_seq_pts_locked(
        self,
        seq_map: dict[int, float],
        raw_items: List[tuple[int, str, float, Optional[float]]],
        avg_dur: float,
        now_ts: float,
    ) -> List[float]:
        n = len(raw_items)
        first_seq = raw_items[0][0]
        # Detect massive sequence discontinuity (> 60 segments jump) and re-anchor cleanly
        if seq_map:
            nearest = min(seq_map.keys(), key=lambda k: abs(k - first_seq))
            if abs(first_seq - nearest) > 60:
                seq_map.clear()

        pts_list = [0.0] * n
        anchor_idx: Optional[int] = None
        for i, (sq, _, _, _) in enumerate(raw_items):
            if sq in seq_map:
                anchor_idx = i
                pts_list[i] = seq_map[sq]
                break

        if anchor_idx is not None:
            for i in range(anchor_idx + 1, n):
                pts_list[i] = pts_list[i - 1] + raw_items[i - 1][2]
            for i in range(anchor_idx - 1, -1, -1):
                pts_list[i] = pts_list[i + 1] - raw_items[i][2]
        elif seq_map:
            max_sq = max(seq_map.keys())
            min_sq = min(seq_map.keys())
            if first_seq > max_sq:
                gap = first_seq - max_sq
                pts_list[0] = seq_map[max_sq] + gap * avg_dur
                for i in range(1, n):
                    pts_list[i] = pts_list[i - 1] + raw_items[i - 1][2]
            else:
                last_sq = raw_items[-1][0]
                gap = min_sq - last_sq
                pts_list[-1] = seq_map[min_sq] - max(1, gap) * avg_dur
                for i in range(n - 2, -1, -1):
                    pts_list[i] = pts_list[i + 1] - raw_items[i][2]
        else:
            # First manifest for this clock: anchor upstream live edge (end of last segment) to now_ts
            pts_list[-1] = now_ts - raw_items[-1][2]
            for i in range(n - 2, -1, -1):
                pts_list[i] = pts_list[i + 1] - raw_items[i][2]

        for i, (sq, _, _, _) in enumerate(raw_items):
            seq_map[sq] = pts_list[i]
        if len(seq_map) > 320:
            cutoff = max(seq_map.keys()) - 260
            for k in [k for k in seq_map if k < cutoff]:
                seq_map.pop(k, None)
        return pts_list

    def ingest_manifest(
        self,
        manifest_url: str,
        manifest_text: str,
        now_wall: Optional[float] = None,
    ) -> List[HlsSegmentMeta]:
        now_ts = now_wall if now_wall is not None else time.time()
        media_seq = 0
        pending_dur: Optional[float] = None
        pending_pdt: Optional[float] = None
        has_any_pdt = False
        raw_items: List[tuple[int, str, float, Optional[float]]] = []
        init_paths_found: List[str] = []

        for raw_line in manifest_text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("#"):
                up = line.upper()
                if up.startswith("#EXT-X-MEDIA-SEQUENCE:"):
                    try:
                        media_seq = int(line.split(":", 1)[1].strip())
                    except ValueError:
                        pass
                elif up.startswith("#EXT-X-MAP:") and 'URI="' in line:
                    m = _URI_ATTR_RE.search(line)
                    if m:
                        abs_init = _unwrap_beacon_url(urljoin(manifest_url, m.group(1)))
                        init_paths_found.append(_norm_seg_path(abs_init))
                elif up.startswith("#EXT-X-PROGRAM-DATE-TIME:"):
                    ep = _parse_iso_utc_epoch(line.split(":", 1)[1])
                    if ep is not None:
                        pending_pdt = ep
                        has_any_pdt = True
                elif up.startswith("#EXTINF:"):
                    val_s = line.split(":", 1)[1].split(",", 1)[0].strip()
                    try:
                        pending_dur = max(0.2, float(val_s))
                    except ValueError:
                        pending_dur = 4.0
                continue

            if pending_dur is None:
                continue
            abs_seg = _unwrap_beacon_url(urljoin(manifest_url, line))
            seg_seq = media_seq + len(raw_items)
            raw_items.append((seg_seq, abs_seg, pending_dur, pending_pdt))
            pending_dur = None
            pending_pdt = None

        if not raw_items and not init_paths_found:
            return []

        with self._lock:
            for ip in init_paths_found:
                self._init_paths.add(ip)
            if not raw_items:
                return []

            n = len(raw_items)
            avg_dur = sum(it[2] for it in raw_items) / n
            pts_list = [0.0] * n

            if has_any_pdt:
                self.uses_pdt = True
                first_pdt_idx = next(i for i, it in enumerate(raw_items) if it[3] is not None)
                pts_list[first_pdt_idx] = raw_items[first_pdt_idx][3] or now_ts
                for i in range(first_pdt_idx + 1, n):
                    if raw_items[i][3] is not None:
                        pts_list[i] = raw_items[i][3] or pts_list[i - 1]
                    else:
                        pts_list[i] = pts_list[i - 1] + raw_items[i - 1][2]
                for i in range(first_pdt_idx - 1, -1, -1):
                    pts_list[i] = pts_list[i + 1] - raw_items[i][2]
            else:
                first_seq = raw_items[0][0]
                use_shared = False
                if not self._shared_seq_pts:
                    use_shared = True
                    self._shared_avg_dur = avg_dur
                else:
                    nearest_sq = min(self._shared_seq_pts.keys(), key=lambda k: abs(k - first_seq))
                    if abs(first_seq - nearest_sq) <= 25 and abs(avg_dur - self._shared_avg_dur) <= 0.85:
                        use_shared = True
                if use_shared:
                    pts_list = self._assign_seq_pts_locked(
                        self._shared_seq_pts, raw_items, avg_dur, now_ts
                    )
                else:
                    v_key = _norm_seg_path(manifest_url)
                    v_map = self._variant_seq_pts.setdefault(v_key, {})
                    pts_list = self._assign_seq_pts_locked(v_map, raw_items, avg_dur, now_ts)

            metas: List[HlsSegmentMeta] = []
            for i, (sq, abs_seg, dur, _) in enumerate(raw_items):
                npath = _norm_seg_path(abs_seg)
                meta = HlsSegmentMeta(
                    seq=sq,
                    url=abs_seg,
                    norm_path=npath,
                    start_pts=round(pts_list[i], 3),
                    duration=round(dur, 3),
                )
                self._url_to_seg[abs_seg] = meta
                self._url_to_seg.move_to_end(abs_seg)
                self._path_to_seg[npath] = meta
                self._path_to_seg.move_to_end(npath)
                metas.append(meta)

            while len(self._url_to_seg) > 512:
                self._url_to_seg.popitem(last=False)
            while len(self._path_to_seg) > 512:
                self._path_to_seg.popitem(last=False)
            return metas


def apply_m3u8_holdback(
    manifest_text: str,
    target_lead_sec: float = 9.5,
    min_keep_segments: int = 4,
) -> tuple[str, int]:
    """
    RFC 8216 compliant zero-transcode M3U8 tail hold-back gate for short-segment live channels:
    - When `seg_dur < 4.5s` (e.g. 2s segments where default TV buffer ~4.4s is shorter than a 6-7s
      spoken English sentence + AI translation), safely trims `K` newest tail segments from the M3U8
      served to the TV while ALWAYS preserving `>= min_keep_segments` (4) segments and keeping
      `#EXT-X-MEDIA-SEQUENCE` 100% valid.
    - Adds 0ms channel-switching penalty and zero CDN expiration risk because held-back segments
      remain live inside the upstream sliding window while NAS FFmpeg taps the untrimmed live edge.
    """
    upper_all = manifest_text.upper()
    if (
        "#EXTINF" not in upper_all
        or "#EXT-X-ENDLIST" in upper_all
        or "PLAYLIST-TYPE:VOD" in upper_all
    ):
        return manifest_text, 0

    header_lines: List[str] = []
    pending_seg_lines: List[str] = []
    pending_dur: Optional[float] = None
    segment_blocks: List[tuple[float, List[str]]] = []

    _PER_SEG_PREFIXES = (
        "#EXTINF:",
        "#EXT-X-PROGRAM-DATE-TIME:",
        "#EXT-X-BYTERANGE:",
        "#EXT-X-DISCONTINUITY",
        "#EXT-X-DATERANGE:",
        "#EXT-X-CUE-",
        "#EXT-X-KEY:",
    )

    for raw_line in manifest_text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            up = stripped.upper()
            if any(up.startswith(pfx) for pfx in _PER_SEG_PREFIXES) or pending_seg_lines:
                pending_seg_lines.append(stripped)
                if up.startswith("#EXTINF:"):
                    val_s = stripped.split(":", 1)[1].split(",", 1)[0].strip()
                    try:
                        pending_dur = max(0.2, float(val_s))
                    except ValueError:
                        pending_dur = 4.0
            elif not segment_blocks:
                header_lines.append(stripped)
            else:
                pending_seg_lines.append(stripped)
            continue

        if pending_dur is not None:
            block = list(pending_seg_lines) + [stripped]
            segment_blocks.append((pending_dur, block))
            pending_seg_lines = []
            pending_dur = None
        else:
            if not segment_blocks:
                header_lines.append(stripped)
            else:
                segment_blocks[-1][1].append(stripped)

    total_segs = len(segment_blocks)
    if total_segs <= min_keep_segments:
        return manifest_text, 0

    avg_dur = sum(d for d, _ in segment_blocks) / total_segs
    if avg_dur >= 4.5:
        return manifest_text, 0

    natural_lead_sec = 2.2 * avg_dur
    deficit_sec = target_lead_sec - natural_lead_sec
    if deficit_sec <= 0.4:
        return manifest_text, 0

    needed_k = int(math.ceil(deficit_sec / max(0.5, avg_dur)))
    trim_k = min(needed_k, total_segs - min_keep_segments, 8)
    if trim_k <= 0:
        return manifest_text, 0

    out_lines: List[str] = list(header_lines)
    for _, blk in segment_blocks[:-trim_k]:
        out_lines.extend(blk)
    return "\n".join(out_lines) + "\n", trim_k


@dataclass
class ClientPlayheadTracker:
    """
    Per-TV closed-loop HLS playhead estimator driven by actual `/seg/{seg_name}` completion timestamps:
    - Anchors playback start `pts_anchor` to the first requested segment once the initial ~2.0s
      startup pre-buffer has been delivered.
    - Keeps playhead advancing smoothly at 1.0x during normal forward pre-fetches (`Playhead <= pts`).
    - Automatically shifts `wall_anchor` forward by the exact stall duration whenever network underrun
      causes a segment to finish arriving AFTER the predicted playhead (`Playhead > pts`, causality bound).
    """

    client_key: str
    pts_anchor: float = 0.0
    wall_anchor: float = 0.0
    first_pts: float = 0.0
    accum_startup_dur: float = 0.0
    last_seq: int = -1
    last_pts: float = 0.0
    last_dur: float = 0.0
    last_done_wall: float = 0.0
    stall_count: int = 0
    calibrated: bool = False
    last_delivered_payload: Optional[dict] = None
    last_delivered_ts: float = 0.0
    last_broadcast_ts: float = 0.0

    def is_fresh(self, now_wall: float, max_idle_sec: float = 30.0) -> bool:
        return (
            self.calibrated
            and self.last_done_wall > 0.0
            and (now_wall - self.last_done_wall) <= max_idle_sec
        )

    def predict_playhead_pts(self, at_wall: float) -> float:
        return self.pts_anchor + (at_wall - self.wall_anchor)

    def predict_wall_time(self, target_pts: float) -> float:
        return self.wall_anchor + (target_pts - self.pts_anchor)

    def on_segment_delivered(self, pts: float, dur: float, seq: int, done_wall: float) -> None:
        safe_dur = max(0.2, dur)
        is_discontinuity = (
            not self.calibrated
            or (
                seq >= 0
                and self.last_seq >= 0
                and (seq < self.last_seq or seq > self.last_seq + 3)
            )
            or (
                self.last_done_wall > 0.0
                and (done_wall - self.last_done_wall) > max(22.0, safe_dur * 5.5)
            )
        )
        if is_discontinuity:
            self.first_pts = pts
            self.pts_anchor = pts
            self.wall_anchor = done_wall + 0.12
            self.accum_startup_dur = safe_dur
            self.calibrated = True
        elif seq >= 0 and seq == self.last_seq:
            # Range / retry fetch of same segment: shift only if causality was violated
            if self.predict_playhead_pts(done_wall) > pts + 0.15:
                self.pts_anchor = pts
                self.wall_anchor = done_wall + 0.08
                self.stall_count += 1
        else:
            if self.accum_startup_dur < 2.0:
                # Startup pre-buffer (<2.0s) still filling: playback of first_pts starts once filled
                self.accum_startup_dur += safe_dur
                self.pts_anchor = self.first_pts
                self.wall_anchor = max(self.wall_anchor, done_wall + 0.10)
            else:
                cur_playhead = self.predict_playhead_pts(done_wall)
                if cur_playhead > pts + 0.08:
                    # Physical causality violation: player buffer ran dry waiting for this segment!
                    self.pts_anchor = pts
                    self.wall_anchor = done_wall + 0.08
                    self.stall_count += 1
                elif (pts + safe_dur) - cur_playhead > max(28.0, safe_dur * 5.5):
                    target_buf = min(12.0, max(3.5, safe_dur * 2.2))
                    self.pts_anchor = (pts + safe_dur) - target_buf
                    self.wall_anchor = done_wall

        self.last_seq = seq
        self.last_pts = pts
        self.last_dur = safe_dur
        self.last_done_wall = done_wall


def rewrite_m3u8_manifest(
    manifest_text: str,
    final_upstream_url: str,
    base_proxy_url: str,
    channel_id: str,
    channel_name: str,
    proxy_segments: bool = False,
    client_id: str = "",
    timeline_clock: Optional[HlsTimelineClock] = None,
) -> str:
    base = base_proxy_url.rstrip("/")
    encoded_name = quote(channel_name)
    clean_cid = sanitize_client_id(client_id)
    cid_param = f"&client={quote(clean_cid)}" if clean_cid else ""
    out_lines: List[str] = []

    def _wrap_playlist(target_url: str) -> str:
        tok = encode_url_token(target_url)
        return f"{base}/live/{channel_id}/index.m3u8?u={tok}&name={encoded_name}{cid_param}"

    def _wrap_segment(
        target_url: str,
        pts: Optional[float] = None,
        dur: Optional[float] = None,
        seq: Optional[int] = None,
        is_init: bool = False,
    ) -> str:
        tok = encode_url_token(target_url)
        seg_name = _safe_seg_filename(target_url, "init.mp4" if is_init else "seg.ts")
        extra = "&init=1" if is_init else ""
        if not is_init:
            if pts is not None:
                extra += f"&pts={pts:.3f}"
            if dur is not None:
                extra += f"&dur={dur:.3f}"
            if seq is not None and seq >= 0:
                extra += f"&seq={seq}"
        return f"{base}/live/{channel_id}/seg/{seg_name}?u={tok}{extra}&name={encoded_name}{cid_param}"

    lines = manifest_text.splitlines()
    expect_variant_playlist = False
    media_seq = 0
    seg_counter = 0
    running_inline_pts = 0.0
    pending_extinf_dur: Optional[float] = None

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            out_lines.append("")
            continue

        if line.startswith("#"):
            upper_line = line.upper()
            if upper_line.startswith("#EXT-X-STREAM-INF"):
                expect_variant_playlist = True
            elif upper_line.startswith("#EXT-X-MEDIA-SEQUENCE:"):
                try:
                    media_seq = int(line.split(":", 1)[1].strip())
                    running_inline_pts = float(media_seq) * 4.0
                except ValueError:
                    pass
            elif upper_line.startswith("#EXTINF:"):
                val_s = line.split(":", 1)[1].split(",", 1)[0].strip()
                try:
                    pending_extinf_dur = max(0.2, float(val_s))
                except ValueError:
                    pending_extinf_dur = 4.0

            if 'URI="' in line:
                is_media_tag = upper_line.startswith("#EXT-X-MEDIA")
                is_map_tag = upper_line.startswith("#EXT-X-MAP")
                is_subtitle_media = is_media_tag and "TYPE=SUBTITLES" in upper_line

                def _replace_uri(match: re.Match) -> str:
                    orig_uri = match.group(1)
                    abs_uri = _unwrap_beacon_url(urljoin(final_upstream_url, orig_uri))
                    if is_subtitle_media:
                        return f'URI="{abs_uri}"'
                    if is_media_tag or abs_uri.split("?")[0].lower().endswith(".m3u8"):
                        return f'URI="{_wrap_playlist(abs_uri)}"'
                    if is_map_tag and proxy_segments:
                        return f'URI="{_wrap_segment(abs_uri, is_init=True)}"'
                    return f'URI="{abs_uri}"'

                line = _URI_ATTR_RE.sub(_replace_uri, line)
            out_lines.append(line)
            continue

        abs_url = _unwrap_beacon_url(urljoin(final_upstream_url, line))
        lower_path = abs_url.split("?")[0].lower()
        is_sub_m3u8 = expect_variant_playlist or lower_path.endswith(".m3u8")
        expect_variant_playlist = False

        if is_sub_m3u8:
            out_lines.append(_wrap_playlist(abs_url))
        elif proxy_segments:
            cur_seq = media_seq + seg_counter
            seg_counter += 1
            cur_dur = pending_extinf_dur if pending_extinf_dur is not None else 4.0
            pending_extinf_dur = None
            meta = timeline_clock.lookup_segment(abs_url) if timeline_clock is not None else None
            if meta is not None:
                seg_pts, seg_dur, seg_sq = meta.start_pts, meta.duration, meta.seq
            else:
                seg_pts, seg_dur, seg_sq = running_inline_pts, cur_dur, cur_seq
            running_inline_pts += cur_dur
            out_lines.append(_wrap_segment(abs_url, pts=seg_pts, dur=seg_dur, seq=seg_sq))
        else:
            pending_extinf_dur = None
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
    sync_mode: str = "onset_clock"
    holdback_segments: int = 0


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
    """Safely kill an FFmpeg subprocess, close its stdout/stderr pipes, and reap it so no zombie remains."""
    if proc is None:
        return
    try:
        proc.kill()
    except Exception:
        pass
    for pipe in (proc.stdout, proc.stderr):
        try:
            if pipe is not None:
                pipe.close()
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
        on_idle_stop: Optional[Callable[[], Awaitable[None]]] = None,
        broadcast_ex_cb: Optional[Callable[[dict, set[str]], Awaitable[None]]] = None,
        send_client_cb: Optional[Callable[[str, dict], Awaitable[None]]] = None,
    ) -> None:
        self.pipeline = pipeline
        self.broadcast_cb = broadcast_cb
        self.broadcast_ex_cb = broadcast_ex_cb
        self.send_client_cb = send_client_cb
        self.idle_timeout_sec = idle_timeout_sec
        self.on_idle_stop = on_idle_stop
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
        self.holdback_segments: int = 0
        self._last_sync_mode: str = "onset_clock"

        # Closed-loop HLS PTS synchronization state
        self.timeline_clock = HlsTimelineClock()
        self._client_playheads: dict[str, ClientPlayheadTracker] = {}
        self._audio_origin_pts: Optional[float] = None
        self._audio_origin_locked: bool = False
        self._tap_first_sample_wall: float = 0.0
        self._tap_last_chunk_wall: float = 0.0
        self._tap_audio_fed_sec: float = 0.0

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

    def record_client_segment(
        self,
        client_key: str,
        pts: float,
        dur: float,
        seq: int = -1,
        done_wall: Optional[float] = None,
    ) -> ClientPlayheadTracker:
        ts = done_wall if done_wall is not None else time.time()
        with self._lock:
            self.last_touch_time = max(self.last_touch_time, ts)
            tracker = self._client_playheads.get(client_key)
            if tracker is None:
                tracker = ClientPlayheadTracker(client_key=client_key)
                self._client_playheads[client_key] = tracker
            tracker.on_segment_delivered(pts=pts, dur=dur, seq=seq, done_wall=ts)
            return tracker

    def remove_client_playhead(self, client_key: str) -> None:
        with self._lock:
            self._client_playheads.pop(client_key, None)

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
        emit_clear: bool = True,
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

            prev_ch_key = (
                canonical_channel_key(self.active_channel_name, self.active_channel_id)
                if (self.active_channel_name or self.active_channel_id)
                else ""
            )
            new_ch_key = canonical_channel_key(channel_name, channel_id)
            old_worker = self._worker_thread
            self._stop_worker_locked()
            session_epoch = self._session_epoch
            self.active_channel_id = channel_id
            self.active_channel_name = channel_name or "Live Channel"
            self.active_upstream_url = dedup_candidates[0] if dedup_candidates else upstream_url
            self.active_candidate_urls = dedup_candidates
            self._last_subtitle_payload = None
            self.holdback_segments = 0
            if prev_ch_key and prev_ch_key != new_ch_key:
                self.timeline_clock = HlsTimelineClock()
                self._client_playheads.clear()

            # Allocate a brand-new Event and Queue per channel session so the stopping
            # worker threads never observe a cleared _stop_event or drained poison pill.
            session_stop = threading.Event()
            session_queue: queue.Queue[Optional[SubtitleEvent]] = queue.Queue(maxsize=32)
            self._stop_event = session_stop
            self._translate_queue = session_queue

            # Clear any previous channel's subtitle on the TV screen immediately
            if emit_clear:
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

    def close(self) -> None:
        """Stops FFmpeg/worker threads and cancels the per-session watchdog task."""
        self.stop_current()
        if self._watchdog_task is not None and not self._watchdog_task.done():
            self._watchdog_task.cancel()

    def _stop_worker_locked(self) -> None:
        self._session_epoch += 1
        self._stop_event.set()
        self._last_subtitle_payload = None
        self._last_subtitle_content_ts = 0.0
        self._audio_origin_pts = None
        self._audio_origin_locked = False
        self._tap_first_sample_wall = 0.0
        self._tap_last_chunk_wall = 0.0
        self._tap_audio_fed_sec = 0.0
        try:
            self._translate_queue.put_nowait(None)
        except queue.Full:
            pass
        proc = self._ffmpeg_proc
        self._ffmpeg_proc = None
        _terminate_ffmpeg_proc(proc)

    def update_hls_manifest_metrics_from_seg_dur(self, seg_dur: float) -> None:
        """
        Calculates fallback stream lead (ms) before closed-loop `/seg/` playhead calibration kicks in:
        - Short segments (< 4.5s) use M3U8 tail hold-back targeting ~9.5s lead.
        - Long segments (>= 4.5s, e.g. Al Jazeera 6s/10s) start FFmpeg at -live_start_index -2
          while TV players start around -3 (~1.2x segment duration lead).
        """
        if seg_dur >= 4.5:
            estimated_lead = int(round(seg_dur * 1200.0))
        else:
            estimated_lead = int(round(max(4500.0, seg_dur * 2200.0)))
        self.auto_stream_lead_ms = max(2800, min(20000, estimated_lead))

    def update_hls_manifest_metrics(self, manifest_text: str) -> None:
        seg_dur = extract_hls_segment_duration(manifest_text)
        if seg_dur is not None:
            self.update_hls_manifest_metrics_from_seg_dur(seg_dur)

    def get_status(self) -> SidecarStatus:
        now = time.time()
        with self._lock:
            is_alive = self._worker_thread is not None and self._worker_thread.is_alive()
            ago = round(now - self.last_touch_time, 1) if self.last_touch_time > 0 else -1.0
            if getattr(self.pipeline, "use_qwen3_asr", False):
                asr_name = "Qwen3-ASR-0.6B"
            elif self.pipeline.use_sensevoice:
                asr_name = "SenseVoice-Small"
            else:
                asr_name = "Zipformer-En"
            llm_tr = self.pipeline.llm_translator
            tr_name = llm_tr.config.model if llm_tr.is_llm_ready else "CTranslate2"
            has_fresh_pts = (
                self._audio_origin_pts is not None
                and any(t.is_fresh(now) for t in self._client_playheads.values())
            )
            if not self.auto_sync:
                smode = "manual"
            elif has_fresh_pts or self._last_sync_mode == "pts_closed_loop":
                smode = "pts_closed_loop" if has_fresh_pts else "onset_clock"
            else:
                smode = "onset_clock"
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
                sync_mode=smode,
                holdback_segments=self.holdback_segments if is_alive else 0,
            )

    async def _idle_watchdog_loop(self) -> None:
        while True:
            await asyncio.sleep(2.5)
            keepalive_payload: Optional[dict] = None
            keepalive_epoch: int = -1
            per_client_keepalives: List[tuple[str, dict]] = []
            calibrated_keys: set[str] = set()
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
                    self._client_playheads.clear()
                    is_alive = False
                    send_clear = True

                if is_alive:
                    keepalive_epoch = self._session_epoch
                    if self.send_client_cb is not None:
                        for ck, tr in self._client_playheads.items():
                            if tr.is_fresh(now):
                                calibrated_keys.add(ck)
                                if (
                                    tr.last_delivered_payload is not None
                                    and (now - tr.last_broadcast_ts) >= 4.0
                                    and (now - tr.last_delivered_ts) <= 38.0
                                ):
                                    per_client_keepalives.append((ck, dict(tr.last_delivered_payload)))
                                    tr.last_broadcast_ts = now

                    if self._last_subtitle_payload is not None:
                        if (
                            (now - self._last_broadcast_ts) >= 4.0
                            and (now - self._last_subtitle_content_ts) <= 38.0
                        ):
                            keepalive_payload = dict(self._last_subtitle_payload)
                            self._last_broadcast_ts = now

            if send_clear:
                await self.broadcast_cb({"type": "clear"})
                if self.on_idle_stop is not None:
                    try:
                        await self.on_idle_stop()
                    except Exception:
                        pass
            else:
                with self._lock:
                    still_valid = self._session_epoch == keepalive_epoch
                if still_valid:
                    if self.send_client_cb is not None:
                        for ck, cp in per_client_keepalives:
                            await self.send_client_cb(ck, cp)
                    if keepalive_payload is not None:
                        if self.broadcast_ex_cb is not None and calibrated_keys:
                            await self.broadcast_ex_cb(keepalive_payload, calibrated_keys)
                        else:
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
                start_idx = "-2" if (detected_seg_dur and detected_seg_dur >= 4.5) else "-1"

                # Reset 16kHz PCM sample cursor for this FFmpeg spawn and pre-ingest audio M3U8 into timeline_clock
                self.pipeline.reset_audio_clock()
                provisional_origin_pts: Optional[float] = None
                if is_hls and is_http:
                    try:
                        hdrs = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) LinguaRelayTV/2.4"}
                        c_opts: dict = {"follow_redirects": True, "timeout": 3.5, "verify": False}
                        if self.stream_proxy:
                            c_opts["proxy"] = self.stream_proxy
                        with httpx.Client(**c_opts) as hc:
                            r_m = hc.get(target_url, headers=hdrs)
                            if r_m.status_code < 400 and "#EXTINF" in r_m.text.upper():
                                target_url = str(r_m.url)
                                segs = self.timeline_clock.ingest_manifest(target_url, r_m.text)
                                if detected_seg_dur is None:
                                    detected_seg_dur = extract_hls_segment_duration(r_m.text)
                                    if detected_seg_dur is not None:
                                        self.update_hls_manifest_metrics_from_seg_dur(detected_seg_dur)
                                        start_idx = "-2" if detected_seg_dur >= 4.5 else "-1"
                                provisional_origin_pts = self.timeline_clock.expected_ffmpeg_start_pts(
                                    segs, int(start_idx)
                                )
                    except Exception:
                        pass

                with self._lock:
                    if ev_stop.is_set():
                        return
                    self.active_upstream_url = target_url
                    self._audio_origin_pts = provisional_origin_pts
                    self._audio_origin_locked = False
                    self._tap_first_sample_wall = 0.0
                    self._tap_last_chunk_wall = 0.0
                    self._tap_audio_fed_sec = 0.0

                cmd = [
                    ffmpeg_bin,
                    "-hide_banner",
                    "-loglevel",
                    "info" if is_hls else "error",
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
                        stderr=subprocess.PIPE if is_hls else subprocess.DEVNULL,
                        bufsize=6400,
                    )
                    with self._lock:
                        if ev_stop.is_set():
                            _terminate_ffmpeg_proc(proc)
                            return
                        self._ffmpeg_proc = proc

                    chunk_size = 3200  # 100ms of 16kHz 16-bit mono PCM (32000 bytes/sec)
                    if is_hls and detected_seg_dur:
                        avail_cdn_sec = detected_seg_dur * (2.0 if start_idx == "-2" else 1.0)
                        lookahead_sec = min(8.0, max(3.5, avail_cdn_sec * 0.80))
                    else:
                        lookahead_sec = 3.5
                    stall_timeout_sec = max(8.5, (detected_seg_dur or 4.0) * 1.6)
                    audio_fed_sec = 0.0
                    tap_start_perf = 0.0
                    proc_spawn_perf = time.perf_counter()
                    last_chunk_perf = proc_spawn_perf
                    win_t0 = 0.0
                    win_audio0 = 0.0
                    slow_warned = False

                    if is_hls and proc.stderr is not None:
                        stderr_pipe = proc.stderr

                        def _stderr_reader(
                            epipe=stderr_pipe,
                            m3u8_url: str = target_url,
                            stop_ev: threading.Event = ev_stop,
                        ) -> None:
                            locked_first = False
                            refreshed_once = False
                            try:
                                while not stop_ev.is_set():
                                    raw_b = epipe.readline()
                                    if not raw_b:
                                        break
                                    if locked_first:
                                        # Keep draining stderr until EOF so pipe buffer never blocks FFmpeg
                                        continue
                                    line_s = raw_b.decode("utf-8", errors="ignore")
                                    m_open = _HLS_OPENING_RE.search(line_s)
                                    if not m_open:
                                        continue
                                    opened_u = m_open.group(1).strip()
                                    low_u = opened_u.split("?")[0].lower()
                                    if low_u.endswith(".m3u8") or self.timeline_clock.is_init_segment(opened_u):
                                        continue
                                    pts = self.timeline_clock.lookup_segment_pts(opened_u)
                                    if pts is None and not refreshed_once and is_http:
                                        refreshed_once = True
                                        try:
                                            c_o: dict = {"follow_redirects": True, "timeout": 2.5, "verify": False}
                                            if self.stream_proxy:
                                                c_o["proxy"] = self.stream_proxy
                                            with httpx.Client(**c_o) as hc2:
                                                r2 = hc2.get(m3u8_url)
                                                if r2.status_code < 400 and "#EXTINF" in r2.text.upper():
                                                    self.timeline_clock.ingest_manifest(str(r2.url), r2.text)
                                                    pts = self.timeline_clock.lookup_segment_pts(opened_u)
                                        except Exception:
                                            pass
                                    if pts is not None:
                                        with self._lock:
                                            self._audio_origin_pts = pts
                                            self._audio_origin_locked = True
                                        locked_first = True
                            except Exception:
                                pass

                        threading.Thread(
                            target=_stderr_reader,
                            daemon=True,
                            name=f"FFmpegLog-{channel_id}",
                        ).start()

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
                            if tap_start_perf == 0.0 and (now_p - proc_spawn_perf) > 8.5:
                                print(
                                    f"[Sidecar] First-audio timeout (>8.5s) on [{channel_name}] ({target_url}), failing over..."
                                )
                                if len(candidates) > 1:
                                    with self._lock:
                                        self._slow_urls.add(raw_url)
                                break
                            if tap_start_perf > 0.0 and (now_p - last_chunk_perf) > stall_timeout_sec:
                                print(
                                    f"[Sidecar] Audio stream stalled (>{stall_timeout_sec:.1f}s) on [{channel_name}] ({target_url}), failing over..."
                                )
                                if len(candidates) > 1:
                                    with self._lock:
                                        self._slow_urls.add(raw_url)
                                break
                            continue

                        if pcm_chunk is None:
                            break

                        now_p = time.perf_counter()
                        now_wall = time.time()
                        last_chunk_perf = now_p
                        if tap_start_perf == 0.0:
                            tap_start_perf = now_p
                            win_t0 = now_p
                            win_audio0 = 0.0
                            with self._lock:
                                self._tap_first_sample_wall = now_wall
                        audio_fed_sec += len(pcm_chunk) / 32000.0
                        with self._lock:
                            self._tap_last_chunk_wall = now_wall
                            self._tap_audio_fed_sec = audio_fed_sec

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
        now_wall = time.time()
        with self._lock:
            if session_epoch is not None and session_epoch != self._session_epoch:
                return
            self.total_subtitles_emitted += 1
            auto_sync = self.auto_sync
            sub_delay = self.subtitle_delay_ms
            lead_ms = self.auto_stream_lead_ms
            disp_mode = self.display_mode
            f_scale = self.font_scale
            origin_pts = self._audio_origin_pts
            tap_first_wall = self._tap_first_sample_wall
            tap_last_wall = self._tap_last_chunk_wall
            tap_fed_sec = self._tap_audio_fed_sec
            fresh_playheads = [
                (ck, tr.predict_wall_time(origin_pts + ev.audio_onset_sec), tr.last_done_wall)
                for ck, tr in self._client_playheads.items()
                if origin_pts is not None
                and ev.audio_onset_sec is not None
                and tr.is_fresh(now_wall)
            ]

        fine_tune_ms = max(-5000, min(8000, sub_delay - 800))
        per_client_schedules: List[tuple[str, int, int]] = []
        sync_mode = "manual"

        if not auto_sync:
            effective_delay_ms = max(0, sub_delay)
            effective_lead_ms = lead_ms
        elif fresh_playheads:
            # Layer 2 Closed-Loop PTS Alignment: schedule each TV at its exact W_tv(onset_pts)
            sync_mode = "pts_closed_loop"
            fresh_playheads.sort(key=lambda item: item[2], reverse=True)
            for ck, pred_wall, _ in fresh_playheads:
                raw_wait_ms = (pred_wall - now_wall) * 1000.0 + fine_tune_ms
                c_delay_ms = max(0, min(28000, int(round(raw_wait_ms))))
                c_lead_ms = max(
                    500,
                    min(35000, int(round((pred_wall - now_wall) * 1000.0 + ev.pipeline_elapsed_ms))),
                )
                per_client_schedules.append((ck, c_delay_ms, c_lead_ms))
            _, effective_delay_ms, effective_lead_ms = per_client_schedules[0]
            with self._lock:
                self.auto_stream_lead_ms = effective_lead_ms
                self._last_sync_mode = sync_mode
        else:
            # Layer 1 Sample-Accurate Speech-Onset Clock Alignment (fallback before /seg/ calibration or HTTP-TS)
            sync_mode = "onset_clock"
            effective_lead_ms = lead_ms
            if ev.audio_onset_sec is not None and tap_first_wall > 0.0:
                tap_base_wall = (
                    max(tap_first_wall, tap_last_wall - tap_fed_sec)
                    if tap_last_wall > 0.0
                    else tap_first_wall
                )
                target_wall = (
                    tap_base_wall
                    + (lead_ms / 1000.0)
                    + ev.audio_onset_sec
                    + (fine_tune_ms / 1000.0)
                )
                effective_delay_ms = max(0, min(25000, int(round((target_wall - now_wall) * 1000.0))))
            elif ev.pipeline_elapsed_ms > 0:
                target_lead_ms = lead_ms + fine_tune_ms
                effective_delay_ms = max(0, int(round(target_lead_ms - ev.pipeline_elapsed_ms)))
            else:
                effective_delay_ms = max(0, sub_delay)
            with self._lock:
                self._last_sync_mode = sync_mode

        def _make_payload(delay_val: int, lead_val: int) -> dict:
            return {
                "type": "subtitle",
                "channel": ev.channel,
                "en": ev.en,
                "zh": ev.zh,
                "is_final": ev.is_final,
                "timestamp": ev.timestamp,
                "latency_ms": ev.latency_ms,
                "pipeline_elapsed_ms": ev.pipeline_elapsed_ms,
                "actual_delay_ms": delay_val,
                "stream_lead_ms": lead_val,
                "auto_sync": auto_sync,
                "sync_mode": sync_mode,
                "audio_onset_sec": ev.audio_onset_sec,
                "engine_info": ev.engine_info,
                "display_mode": disp_mode,
                "font_scale": f_scale,
            }

        default_payload = _make_payload(effective_delay_ms, effective_lead_ms)
        with self._lock:
            if session_epoch is not None and session_epoch != self._session_epoch:
                return
            if ev.is_final:
                self.recent_subtitles.append(default_payload)
                if len(self.recent_subtitles) > 60:
                    self.recent_subtitles = self.recent_subtitles[-60:]

        if per_client_schedules and self.send_client_cb is not None:
            calibrated_keys = {ck for ck, _, _ in per_client_schedules}
            for ck, c_delay, c_lead in per_client_schedules:
                self._schedule_client_delivery(
                    client_key=ck,
                    payload=_make_payload(c_delay, c_lead),
                    delay_ms=c_delay,
                    session_epoch=session_epoch,
                )
            self._schedule_broadcast(
                default_payload,
                delay_ms=effective_delay_ms,
                session_epoch=session_epoch,
                exclude_clients=calibrated_keys,
            )
        else:
            self._schedule_broadcast(
                default_payload, delay_ms=effective_delay_ms, session_epoch=session_epoch
            )

    def _schedule_client_delivery(
        self,
        client_key: str,
        payload: dict,
        delay_ms: int,
        session_epoch: Optional[int] = None,
    ) -> None:
        if self._loop is None or self._loop.is_closed() or self.send_client_cb is None:
            return
        send_cb = self.send_client_cb

        async def _delayed_client_send() -> None:
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
                tr = self._client_playheads.get(client_key)
                if tr is not None and payload.get("type") == "subtitle":
                    tr.last_delivered_payload = payload
                    tr.last_delivered_ts = now
                    tr.last_broadcast_ts = now
            await send_cb(client_key, payload)

        asyncio.run_coroutine_threadsafe(_delayed_client_send(), self._loop)

    def _schedule_broadcast(
        self,
        payload: dict,
        delay_ms: int,
        session_epoch: Optional[int] = None,
        exclude_clients: Optional[set[str]] = None,
    ) -> None:
        if self._loop is None or self._loop.is_closed():
            return
        excl = set(exclude_clients) if exclude_clients else set()

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
            if excl and self.broadcast_ex_cb is not None:
                await self.broadcast_ex_cb(payload, excl)
            else:
                await self.broadcast_cb(payload)

        asyncio.run_coroutine_threadsafe(_delayed_send(), self._loop)


def make_client_key(client_ip: str, client_id: str = "") -> str:
    cid = sanitize_client_id(client_id)
    if cid:
        return f"tag:{cid}"
    ip = (client_ip or "local").strip()
    if ip.startswith("::ffff:"):
        ip = ip[7:]
    return f"ip:{ip}"


def canonical_channel_key(channel_name: str, channel_id: str = "") -> str:
    base = re.sub(r"\s*[\(（]源\d+[\)）]$", "", (channel_name or "").strip()).strip()
    if base and base.lower() not in ("live channel", "manual stream", "live stream"):
        return base.upper()
    return (channel_id or base or "MANUAL").upper()


@dataclass
class ClientBinding:
    client_key: str
    client_ip: str
    client_tag: str
    channel_key: str
    channel_id: str
    channel_name: str
    need_translate: bool
    last_seen: float


@dataclass
class ChannelSessionEntry:
    channel_key: str
    channel_id: str
    channel_name: str
    sidecar: AudioSidecarManager
    viewers: set[str] = field(default_factory=set)
    viewer_last_seen: dict[str, float] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


class MultiSessionSidecarPool:
    """
    Concurrent multi-client sidecar orchestrator (up to `max_sessions` independent foreign channels).
    - Multiple clients watching the SAME foreign channel automatically share 1 FFmpeg + VAD + ASR + LLM session.
    - Multiple clients watching DIFFERENT channels get isolated FFmpeg + Silero-VAD + 6-sentence LLM context
      while sharing the read-only `Qwen3-ASR-0.6B-int8` model weights in memory.
    - A client switching to a Chinese channel (`need_translate=False`) clears ONLY its own screen and never
      stops other clients' active foreign sessions.
    """

    def __init__(
        self,
        pipeline: StreamingSpeechPipeline,
        route_session_cb: Callable[[str, set[str], bool, int, dict], Awaitable[None]],
        send_client_cb: Callable[[str, dict], Awaitable[None]],
        broadcast_all_cb: Callable[[dict], Awaitable[None]],
        max_sessions: Optional[int] = None,
        idle_timeout_sec: float = 25.0,
    ) -> None:
        self.pipeline = pipeline
        self.route_session_cb = route_session_cb
        self.send_client_cb = send_client_cb
        self.broadcast_all_cb = broadcast_all_cb
        env_max = os.environ.get("LINGUARELAY_MAX_SESSIONS", "").strip()
        default_max = int(env_max) if env_max.isdigit() and int(env_max) > 0 else 3
        self.max_sessions: int = max(1, max_sessions if max_sessions is not None else default_max)
        self.idle_timeout_sec: float = idle_timeout_sec
        self.stream_proxy: str = os.environ.get("LINGUARELAY_STREAM_PROXY", "").strip()

        # Shared default display & alignment settings
        self.subtitle_delay_ms: int = 800
        self.display_mode: str = "bilingual"
        self.font_scale: float = 1.0
        self.sentence_mode: str = "full_sentence"
        self.auto_sync: bool = True

        self.recent_subtitles: List[dict] = []
        self._archived_subtitles_count: int = 0
        self._sessions: dict[str, ChannelSessionEntry] = {}
        self._client_bindings: dict[str, ClientBinding] = {}
        self._lock = threading.RLock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._housekeeping_task: Optional[asyncio.Task] = None

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        with self._lock:
            entries = list(self._sessions.values())
        for entry in entries:
            entry.sidecar.attach_loop(loop)
        if self._housekeeping_task is None or self._housekeeping_task.done():
            self._housekeeping_task = loop.create_task(self._pool_housekeeping_loop())

    def close(self) -> None:
        self.stop_current()
        if self._housekeeping_task is not None and not self._housekeeping_task.done():
            self._housekeeping_task.cancel()

    def _sync_settings_to_sidecar(self, sc: AudioSidecarManager) -> None:
        sc.subtitle_delay_ms = self.subtitle_delay_ms
        sc.display_mode = self.display_mode
        sc.font_scale = self.font_scale
        sc.sentence_mode = self.sentence_mode
        sc.pipeline.sentence_mode = self.sentence_mode
        sc.auto_sync = self.auto_sync
        sc.stream_proxy = self.stream_proxy

    def apply_settings(
        self,
        subtitle_delay_ms: Optional[int] = None,
        display_mode: Optional[str] = None,
        font_scale: Optional[float] = None,
        sentence_mode: Optional[str] = None,
        auto_sync: Optional[bool] = None,
    ) -> None:
        with self._lock:
            if subtitle_delay_ms is not None:
                self.subtitle_delay_ms = subtitle_delay_ms
            if display_mode in ("bilingual", "zh_only"):
                self.display_mode = display_mode
            if font_scale is not None:
                self.font_scale = font_scale
            if sentence_mode in ("full_sentence", "paragraph", "fast_clause"):
                self.sentence_mode = sentence_mode
                self.pipeline.sentence_mode = sentence_mode
            if auto_sync is not None:
                self.auto_sync = auto_sync
            entries = list(self._sessions.values())
        for entry in entries:
            self._sync_settings_to_sidecar(entry.sidecar)

    async def _on_session_broadcast(
        self,
        ch_key: str,
        msg: dict,
        exclude_clients: Optional[set[str]] = None,
    ) -> None:
        with self._lock:
            entry = self._sessions.get(ch_key)
            if entry is None:
                return
            viewers = set(entry.viewers)
            manual_pinned = entry.sidecar.manual_pinned
            active_cnt = sum(1 for e in self._sessions.values() if e.sidecar.get_status().active)
            if msg.get("type") == "subtitle" and msg.get("is_final", True):
                self.recent_subtitles.append(msg)
                if len(self.recent_subtitles) > 80:
                    self.recent_subtitles = self.recent_subtitles[-80:]
        if exclude_clients:
            await self.route_session_cb(
                ch_key, viewers, manual_pinned, max(1, active_cnt), msg, exclude_clients
            )
        else:
            await self.route_session_cb(ch_key, viewers, manual_pinned, max(1, active_cnt), msg)

    async def _on_session_idle_stop(self, ch_key: str) -> None:
        with self._lock:
            entry = self._sessions.pop(ch_key, None)
            if entry is not None:
                self._archived_subtitles_count += entry.sidecar.total_subtitles_emitted
                entry.sidecar.close()

    def _spawn_session_locked(
        self,
        ch_key: str,
        channel_id: str,
        channel_name: str,
    ) -> ChannelSessionEntry:
        sess_pipeline = self.pipeline.create_session_pipeline(channel_name=channel_name)
        sc = AudioSidecarManager(
            pipeline=sess_pipeline,
            broadcast_cb=lambda msg, ck=ch_key: self._on_session_broadcast(ck, msg),
            broadcast_ex_cb=lambda msg, excl, ck=ch_key: self._on_session_broadcast(ck, msg, excl),
            send_client_cb=self.send_client_cb,
            idle_timeout_sec=self.idle_timeout_sec,
            on_idle_stop=lambda ck=ch_key: self._on_session_idle_stop(ck),
        )
        self._sync_settings_to_sidecar(sc)
        if self._loop is not None:
            sc.attach_loop(self._loop)
        entry = ChannelSessionEntry(
            channel_key=ch_key,
            channel_id=channel_id,
            channel_name=channel_name,
            sidecar=sc,
        )
        self._sessions[ch_key] = entry
        return entry

    def _evict_one_if_needed_locked(self, incoming_ch_key: str) -> tuple[Optional[ChannelSessionEntry], set[str]]:
        # First reap any dead unpinned entries
        dead_keys = [
            k
            for k, e in self._sessions.items()
            if k != incoming_ch_key and not e.sidecar.get_status().active and not e.sidecar.manual_pinned
        ]
        for k in dead_keys:
            dead = self._sessions.pop(k)
            self._archived_subtitles_count += dead.sidecar.total_subtitles_emitted
            dead.sidecar.close()

        if incoming_ch_key in self._sessions or len(self._sessions) < self.max_sessions:
            return None, set()

        # Pick eviction victim: prefer unpinned with 0 viewers -> unpinned oldest touch -> oldest overall
        candidates = [e for k, e in self._sessions.items() if k != incoming_ch_key]
        if not candidates:
            return None, set()
        candidates.sort(
            key=lambda e: (
                1 if e.sidecar.manual_pinned else 0,
                1 if len(e.viewers) > 0 else 0,
                e.sidecar.last_touch_time,
            )
        )
        victim = candidates[0]
        self._sessions.pop(victim.channel_key, None)
        self._archived_subtitles_count += victim.sidecar.total_subtitles_emitted
        evicted_viewers = set(victim.viewers)
        victim.sidecar.close()
        return victim, evicted_viewers

    async def touch_client_channel(
        self,
        client_key: str,
        client_ip: str,
        client_tag: str,
        channel_id: str,
        channel_name: str,
        upstream_url: str,
        candidate_urls: Optional[List[str]] = None,
        manifest_text: str = "",
    ) -> AudioSidecarManager:
        now = time.time()
        ch_key = canonical_channel_key(channel_name, channel_id)
        send_clear_to_client = False
        evicted_viewers: set[str] = set()
        replay_payload: Optional[dict] = None

        with self._lock:
            prev = self._client_bindings.get(client_key)
            self._client_bindings[client_key] = ClientBinding(
                client_key=client_key,
                client_ip=client_ip,
                client_tag=client_tag,
                channel_key=ch_key,
                channel_id=channel_id,
                channel_name=channel_name,
                need_translate=True,
                last_seen=now,
            )

            if prev is not None and prev.channel_key and prev.channel_key != ch_key:
                send_clear_to_client = True
                old_entry = self._sessions.get(prev.channel_key)
                if old_entry is not None:
                    old_entry.viewers.discard(client_key)
                    old_entry.viewer_last_seen.pop(client_key, None)
                    old_entry.sidecar.remove_client_playhead(client_key)
                    if not old_entry.viewers and not old_entry.sidecar.manual_pinned:
                        self._sessions.pop(prev.channel_key, None)
                        self._archived_subtitles_count += old_entry.sidecar.total_subtitles_emitted
                        old_entry.sidecar.close()

            entry = self._sessions.get(ch_key)
            is_existing = entry is not None and entry.sidecar.get_status().active
            if entry is None:
                _, evicted_viewers = self._evict_one_if_needed_locked(ch_key)
                entry = self._spawn_session_locked(ch_key, channel_id, channel_name)
                send_clear_to_client = True

            was_new_viewer = client_key not in entry.viewers
            if prev is None or prev.channel_key != ch_key:
                send_clear_to_client = True
            entry.viewers.add(client_key)
            entry.viewer_last_seen[client_key] = now
            other_viewers_active = is_existing and len(entry.viewers) > 1
            if not other_viewers_active:
                entry.channel_id = channel_id
                entry.channel_name = channel_name
            sc = entry.sidecar

            if is_existing and was_new_viewer:
                if sc._last_subtitle_payload and (now - sc._last_subtitle_content_ts) <= 35.0:
                    replay_payload = dict(sc._last_subtitle_payload)
                else:
                    replay_payload = {
                        "type": "status",
                        "event": "channel_join",
                        "channel": entry.channel_name,
                        "en": f"Listening: {entry.channel_name}",
                        "zh": f"声幕已接入频道：{entry.channel_name}",
                        "is_final": True,
                        "display_mode": self.display_mode,
                    }

        for ev_ck in evicted_viewers:
            if ev_ck != client_key:
                await self.send_client_cb(ev_ck, {"type": "clear"})
        if send_clear_to_client:
            await self.send_client_cb(client_key, {"type": "clear"})

        if other_viewers_active:
            with sc._lock:
                sc.last_touch_time = time.time()
        else:
            if manifest_text:
                sc.update_hls_manifest_metrics(manifest_text)
            sc.touch_channel(
                channel_id=channel_id,
                channel_name=channel_name,
                upstream_url=upstream_url,
                candidate_urls=candidate_urls,
                emit_clear=False,
            )

        if replay_payload is not None:
            await self.send_client_cb(client_key, replay_payload)
        return sc

    async def mark_client_chinese_channel(
        self,
        client_key: str,
        client_ip: str,
        client_tag: str,
        channel_name: str,
    ) -> None:
        now = time.time()
        closed_last_session = False
        with self._lock:
            prev = self._client_bindings.get(client_key)
            self._client_bindings[client_key] = ClientBinding(
                client_key=client_key,
                client_ip=client_ip,
                client_tag=client_tag,
                channel_key="",
                channel_id="",
                channel_name=channel_name,
                need_translate=False,
                last_seen=now,
            )
            if prev is not None and prev.channel_key:
                old_entry = self._sessions.get(prev.channel_key)
                if old_entry is not None:
                    old_entry.viewers.discard(client_key)
                    old_entry.viewer_last_seen.pop(client_key, None)
                    old_entry.sidecar.remove_client_playhead(client_key)
                    if not old_entry.viewers and not old_entry.sidecar.manual_pinned:
                        self._sessions.pop(prev.channel_key, None)
                        self._archived_subtitles_count += old_entry.sidecar.total_subtitles_emitted
                        old_entry.sidecar.close()
                        if not self._sessions:
                            closed_last_session = True

        await self.send_client_cb(client_key, {"type": "clear"})
        if closed_last_session:
            await self.route_session_cb("", set(), False, 0, {"type": "clear"})

    def _find_session_locked(
        self,
        channel_id: str,
        channel_name: str,
        client_key: str = "",
    ) -> Optional[ChannelSessionEntry]:
        ch_key = canonical_channel_key(channel_name, channel_id)
        entry = self._sessions.get(ch_key)
        if entry is None and channel_id:
            for cand in self._sessions.values():
                if cand.channel_id == channel_id or cand.sidecar.active_channel_id == channel_id:
                    entry = cand
                    break
        if entry is None and client_key:
            b = self._client_bindings.get(client_key)
            if b is not None and b.need_translate and (not channel_id or b.channel_id == channel_id):
                entry = self._sessions.get(b.channel_key)
        return entry

    def prepare_client_media_manifest(
        self,
        channel_id: str,
        channel_name: str,
        final_url: str,
        manifest_text: str,
        client_key: str = "",
    ) -> tuple[str, Optional[HlsTimelineClock]]:
        """
        1) Ingests the untrimmed upstream Media M3U8 into the channel session's `HlsTimelineClock`
           so both held-back TV segments and live-edge FFmpeg segments share an exact PTS mapping.
        2) Applies adaptive M3U8 tail hold-back (`target_lead_sec=9.5`) when `auto_sync` is enabled.
        """
        if "#EXTINF" not in manifest_text.upper():
            return manifest_text, None

        with self._lock:
            entry = self._find_session_locked(channel_id, channel_name, client_key)
            sc = entry.sidecar if entry is not None else None
            use_holdback = self.auto_sync

        clock: Optional[HlsTimelineClock] = None
        if sc is not None:
            clock = sc.timeline_clock
            clock.ingest_manifest(final_url, manifest_text)

        if use_holdback:
            held_text, held_k = apply_m3u8_holdback(manifest_text, target_lead_sec=9.5)
            if sc is not None:
                sc.holdback_segments = held_k
            return held_text, clock

        if sc is not None:
            sc.holdback_segments = 0
        return manifest_text, clock

    def record_client_segment(
        self,
        client_key: str,
        client_ip: str,
        client_tag: str,
        channel_id: str,
        channel_name: str,
        pts: Optional[float],
        dur: Optional[float],
        seq: int = -1,
        done_wall: Optional[float] = None,
    ) -> None:
        ts = done_wall if done_wall is not None else time.time()
        sc: Optional[AudioSidecarManager] = None
        with self._lock:
            entry = self._find_session_locked(channel_id, channel_name, client_key)
            b = self._client_bindings.get(client_key)
            if b is not None and b.need_translate:
                if (entry is not None and b.channel_key == entry.channel_key) or (
                    channel_id and b.channel_id == channel_id
                ):
                    b.last_seen = max(b.last_seen, ts)
            if entry is not None:
                entry.viewers.add(client_key)
                entry.viewer_last_seen[client_key] = ts
                sc = entry.sidecar
        if sc is not None:
            if pts is not None and dur is not None and dur > 0:
                sc.record_client_segment(
                    client_key=client_key, pts=pts, dur=dur, seq=seq, done_wall=ts
                )
            else:
                with sc._lock:
                    sc.last_touch_time = max(sc.last_touch_time, ts)

    async def pin_channel(
        self,
        channel_id: str,
        channel_name: str,
        upstream_url: str,
        candidate_urls: Optional[List[str]] = None,
    ) -> SidecarStatus:
        ch_key = canonical_channel_key(channel_name, channel_id)
        evicted_viewers: set[str] = set()
        with self._lock:
            entry = self._sessions.get(ch_key)
            if entry is None:
                _, evicted_viewers = self._evict_one_if_needed_locked(ch_key)
                entry = self._spawn_session_locked(ch_key, channel_id, channel_name)
            entry.channel_id = channel_id
            entry.channel_name = channel_name
            sc = entry.sidecar

        for ev_ck in evicted_viewers:
            await self.send_client_cb(ev_ck, {"type": "clear"})
        sc.touch_channel(
            channel_id=channel_id,
            channel_name=channel_name,
            upstream_url=upstream_url,
            manual_pin=True,
            candidate_urls=candidate_urls,
        )
        return self.get_status()

    def set_client_delay(self, client_key: str, delay_ms: int) -> None:
        with self._lock:
            binding = self._client_bindings.get(client_key)
            if binding and binding.channel_key and binding.channel_key in self._sessions:
                self._sessions[binding.channel_key].sidecar.subtitle_delay_ms = delay_ms
                if len(self._sessions) <= 1:
                    self.subtitle_delay_ms = delay_ms
            else:
                self.subtitle_delay_ms = delay_ms
                for e in self._sessions.values():
                    e.sidecar.subtitle_delay_ms = delay_ms

    def stop_session(self, channel_key: str = "") -> tuple[bool, set[str]]:
        with self._lock:
            if channel_key:
                entry = self._sessions.pop(channel_key, None)
                if entry is None:
                    # Also allow matching by channel_id or channel_name
                    for k, e in list(self._sessions.items()):
                        if e.channel_id == channel_key or e.channel_name == channel_key:
                            entry = self._sessions.pop(k)
                            break
                if entry is not None:
                    viewers = set(entry.viewers)
                    self._archived_subtitles_count += entry.sidecar.total_subtitles_emitted
                    entry.sidecar.close()
                    return True, viewers
                return False, set()
            all_viewers: set[str] = set()
            for e in self._sessions.values():
                all_viewers.update(e.viewers)
            return self.stop_current(), all_viewers

    def stop_current(self) -> bool:
        with self._lock:
            had_any = bool(self._sessions)
            for entry in self._sessions.values():
                self._archived_subtitles_count += entry.sidecar.total_subtitles_emitted
                entry.sidecar.close()
            self._sessions.clear()
            return had_any

    def get_client_binding(self, client_key: str, max_age_sec: float = 90.0) -> Optional[ClientBinding]:
        now = time.time()
        with self._lock:
            b = self._client_bindings.get(client_key)
            if b is not None and (now - b.last_seen) <= max_age_sec:
                return b
            return None

    async def _pool_housekeeping_loop(self) -> None:
        while True:
            await asyncio.sleep(5.0)
            now = time.time()
            with self._lock:
                # Expire stale viewer references inside sessions
                for ck, entry in list(self._sessions.items()):
                    stale_v = [
                        vk
                        for vk, ts in entry.viewer_last_seen.items()
                        if (now - ts) > (self.idle_timeout_sec + 5.0)
                    ]
                    for vk in stale_v:
                        entry.viewers.discard(vk)
                        entry.viewer_last_seen.pop(vk, None)
                        entry.sidecar.remove_client_playhead(vk)
                    st = entry.sidecar.get_status()
                    if not st.active and not entry.sidecar.manual_pinned:
                        self._sessions.pop(ck, None)
                        self._archived_subtitles_count += entry.sidecar.total_subtitles_emitted
                        entry.sidecar.close()

                # Expire old client bindings (> 120s inactive)
                stale_cb = [
                    ck for ck, b in self._client_bindings.items() if (now - b.last_seen) > 120.0
                ]
                for ck in stale_cb:
                    self._client_bindings.pop(ck, None)

    def get_sessions_summary(self) -> List[dict]:
        now = time.time()
        out: List[dict] = []
        with self._lock:
            entries = sorted(
                self._sessions.values(),
                key=lambda e: e.sidecar.last_touch_time,
                reverse=True,
            )
            for e in entries:
                st = e.sidecar.get_status()
                if not st.active and not e.sidecar.manual_pinned:
                    continue
                out.append(
                    {
                        "channel_key": e.channel_key,
                        "channel_id": st.channel_id or e.channel_id,
                        "channel_name": st.channel_name or e.channel_name,
                        "upstream_url": st.upstream_url,
                        "manual_pinned": st.manual_pinned,
                        "viewer_count": len(e.viewers),
                        "viewers": sorted(e.viewers),
                        "subtitles_emitted": st.total_subtitles_emitted,
                        "auto_stream_lead_ms": st.auto_stream_lead_ms,
                        "sync_mode": st.sync_mode,
                        "holdback_segments": st.holdback_segments,
                        "last_touch_ago_sec": round(max(0.0, now - e.sidecar.last_touch_time), 1)
                        if e.sidecar.last_touch_time > 0
                        else 0.0,
                    }
                )
        return out

    def get_clients_summary(self) -> List[dict]:
        now = time.time()
        out: List[dict] = []
        with self._lock:
            for b in sorted(self._client_bindings.values(), key=lambda x: x.last_seen, reverse=True):
                age = now - b.last_seen
                if age > 90.0:
                    continue
                out.append(
                    {
                        "client_key": b.client_key,
                        "client_ip": b.client_ip,
                        "client_tag": b.client_tag,
                        "channel_name": b.channel_name,
                        "need_translate": b.need_translate,
                        "last_seen_ago_sec": round(max(0.0, age), 1),
                    }
                )
        return out

    def get_status(self) -> SidecarStatus:
        with self._lock:
            active_statuses: List[tuple[ChannelSessionEntry, SidecarStatus]] = []
            live_emitted = 0
            for e in self._sessions.values():
                st = e.sidecar.get_status()
                live_emitted += st.total_subtitles_emitted
                if st.active:
                    active_statuses.append((e, st))

            total_emitted = self._archived_subtitles_count + live_emitted
            if getattr(self.pipeline, "use_qwen3_asr", False):
                asr_name = "Qwen3-ASR-0.6B"
            elif self.pipeline.use_sensevoice:
                asr_name = "SenseVoice-Small"
            else:
                asr_name = "Zipformer-En"
            llm_tr = self.pipeline.llm_translator
            tr_name = llm_tr.config.model if llm_tr.is_llm_ready else "CTranslate2"

            if not active_statuses:
                return SidecarStatus(
                    active=False,
                    channel_id="",
                    channel_name="",
                    upstream_url="",
                    manual_pinned=False,
                    subtitle_delay_ms=self.subtitle_delay_ms,
                    display_mode=self.display_mode,
                    last_touch_ago_sec=-1.0,
                    total_subtitles_emitted=total_emitted,
                    asr_engine=asr_name,
                    translation_engine=tr_name,
                    llm_ready=llm_tr.is_llm_ready,
                    llm_error=llm_tr.last_error,
                    sentence_mode=self.pipeline.sentence_mode,
                    auto_sync=self.auto_sync,
                    auto_stream_lead_ms=3500,
                    sync_mode="manual" if not self.auto_sync else "onset_clock",
                    holdback_segments=0,
                )

            active_statuses.sort(key=lambda pair: pair[0].sidecar.last_touch_time, reverse=True)
            primary_entry, primary_st = active_statuses[0]
            if len(active_statuses) == 1:
                disp_ch_name = primary_st.channel_name
            else:
                names = " / ".join(s.channel_name for _, s in active_statuses)
                disp_ch_name = f"{names} ({len(active_statuses)}路并发)"

            return SidecarStatus(
                active=True,
                channel_id=primary_st.channel_id,
                channel_name=disp_ch_name,
                upstream_url=primary_st.upstream_url,
                manual_pinned=any(s.manual_pinned for _, s in active_statuses),
                subtitle_delay_ms=self.subtitle_delay_ms,
                display_mode=self.display_mode,
                last_touch_ago_sec=primary_st.last_touch_ago_sec,
                total_subtitles_emitted=total_emitted,
                asr_engine=asr_name,
                translation_engine=tr_name,
                llm_ready=llm_tr.is_llm_ready,
                llm_error=llm_tr.last_error,
                sentence_mode=self.pipeline.sentence_mode,
                auto_sync=self.auto_sync,
                auto_stream_lead_ms=primary_st.auto_stream_lead_ms,
                sync_mode=primary_st.sync_mode,
                holdback_segments=primary_st.holdback_segments,
            )
