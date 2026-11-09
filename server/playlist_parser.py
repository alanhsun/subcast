"""
Playlist parser and rewriter for TVBox / 影视仓 (.txt) and Jellyfin (.m3u / .m3u8).
Supports:
- TVBox .txt format ("分组名,#genre#" + "频道名,http://...")
- M3U / M3U8 format ("#EXTINF:-1 group-title=...,频道名" + "http://...")
- Automatic detection of foreign/English channels (`need_translate`) vs Chinese channels
- Rewriting channel URLs to pass through the LinguaRelay-TV sidecar proxy with full EPG metadata.
"""

from __future__ import annotations

import base64
import hashlib
import re
from dataclasses import dataclass
from typing import List
from urllib.parse import quote

# Domestic/Chinese groups where channels are already in Chinese (unless channel name matches foreign keywords)
_CHINESE_GROUPS = {
    "央视频道",
    "卫视频道",
    "地方频道",
    "香港/澳门",
    "台湾",
    "电影频道",
    "纪录频道",
    "儿童频道",
    "数字频道",
    "音乐频道",
    "解说频道",
}

_FOREIGN_NAME_KEYWORDS = (
    "cgtn",
    "bbc",
    "cnn",
    "sky news",
    "skynews",
    "dw",
    "德国之声",
    "cna",
    "亚洲新闻台",
    "channel newsasia",
    "france 24",
    "france24",
    "abc news",
    "global news",
    "gusto",
    "red bull",
    "travelxp",
    "travel xp",
    "wion",
    "yahoo",
    "supreme master",
    "无上师",
    "love nature",
    "history",
    "历史频道",
    "discovery",
    "nat geo",
    "national geographic",
    "bloomberg",
    "cnbc",
    "fox",
    "nbc",
    "cbs",
    "espn",
    "eurosport",
    "al jazeera",
    "nhk world",
    "arirang",
    "kbs world",
)


def is_foreign_channel(name: str, group: str) -> bool:
    """
    Determine whether a channel should trigger the real-time English->Chinese AI subtitle pipeline.
    - All channels in '国际/海外' or non-domestic groups default to True.
    - Any channel whose name matches foreign/English broadcast keywords (e.g., CGTN, Love Nature 4K,
      History, BBC, Sky News, DW, CNA) returns True even if placed in '央视频道' or '4K超高清'.
    """
    low_name = (name or "").lower()
    grp = (group or "").strip()
    if any(kw in low_name for kw in _FOREIGN_NAME_KEYWORDS):
        return True
    if grp in ("国际/海外", "国际直播", "欧美英语直播", "国际频道", "海外频道"):
        return True
    if grp in _CHINESE_GROUPS:
        return False
    # For 4K超高清 or custom groups: check if the name contains ASCII letters other than 4K/8K/HD/CCTV/TVB/TVBS/CHC
    cleaned = re.sub(r"\b(4k|8k|hd|fhd|uhd|cctv|tvb|tvbs|chc)\b", "", low_name, flags=re.I)
    if re.search(r"[a-z]{3,}", cleaned):
        return True
    return grp not in _CHINESE_GROUPS


@dataclass
class ChannelEntry:
    id: str
    name: str
    url: str
    group: str = "国际直播"
    logo: str = ""
    tvg_id: str = ""
    tvg_name: str = ""
    need_translate: bool = True

    @property
    def is_hls(self) -> bool:
        lower = self.url.lower().split("?")[0]
        return lower.endswith(".m3u8") or "m3u8" in self.url.lower()


def _make_channel_id(name: str, url: str) -> str:
    digest = hashlib.sha1(f"{name}|{url}".encode("utf-8")).hexdigest()[:12]
    return digest


def encode_url_token(url: str) -> str:
    return base64.urlsafe_b64encode(url.encode("utf-8")).decode("ascii").rstrip("=")


def decode_url_token(token: str) -> str:
    padding = "=" * ((4 - len(token) % 4) % 4)
    return base64.urlsafe_b64decode((token + padding).encode("ascii")).decode("utf-8")


_ATTR_RE = re.compile(r'([a-zA-Z0-9_-]+)="([^"]*)"')


def parse_playlist(content: str) -> List[ChannelEntry]:
    """Auto-detect and parse either M3U (.m3u/.m3u8) or TVBox (.txt) content."""
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if not lines:
        return []

    is_m3u = any(
        line.upper().startswith("#EXTM3U") or line.upper().startswith("#EXTINF")
        for line in lines[:20]
    )
    if is_m3u:
        return parse_m3u(lines)
    return parse_txt(lines)


def _default_logo_url(name: str) -> str:
    clean = re.sub(r"\s*[\(（][^\)）]*[\)）]\s*", "", name or "").strip()
    clean = re.sub(r"\s+(HD|FHD|UHD|4K|8K)$", "", clean, flags=re.I).strip()
    clean = clean.replace(" ", "")
    if not clean:
        return ""
    return f"https://live.fanmingming.com/tv/{quote(clean)}.png"


def parse_m3u(lines: List[str]) -> List[ChannelEntry]:
    channels: List[ChannelEntry] = []
    current_name = "Unknown Channel"
    current_group = "国际直播"
    current_logo = ""
    current_tvg_id = ""
    current_tvg_name = ""

    for line in lines:
        if line.upper().startswith("#EXTM3U"):
            continue
        if line.upper().startswith("#EXTINF"):
            attrs = dict(_ATTR_RE.findall(line))
            current_group = attrs.get("group-title") or attrs.get("group") or "国际直播"
            current_logo = attrs.get("tvg-logo", "")
            current_tvg_id = attrs.get("tvg-id", "")
            current_tvg_name = attrs.get("tvg-name", "")
            if "," in line:
                current_name = line.rsplit(",", 1)[1].strip() or current_tvg_name or "Channel"
            else:
                current_name = current_tvg_name or "Channel"
        elif line.startswith("#"):
            continue
        elif line.startswith(("http://", "https://", "rtmp://", "rtsp://")):
            cid = _make_channel_id(current_name, line)
            tvg_name = current_tvg_name or current_name
            channels.append(
                ChannelEntry(
                    id=cid,
                    name=current_name,
                    url=line,
                    group=current_group,
                    logo=current_logo or _default_logo_url(tvg_name),
                    tvg_id=current_tvg_id or current_name,
                    tvg_name=tvg_name,
                    need_translate=is_foreign_channel(current_name, current_group),
                )
            )
            current_name = "Unknown Channel"
            current_logo = ""
            current_tvg_id = ""
            current_tvg_name = ""
    return channels


def parse_txt(lines: List[str]) -> List[ChannelEntry]:
    channels: List[ChannelEntry] = []
    current_group = "国际直播"

    for line in lines:
        if line.startswith("#") or line.startswith("//"):
            continue
        if ",#genre#" in line.lower():
            current_group = line.split(",", 1)[0].strip() or "国际直播"
            continue
        if "," not in line:
            continue
        name, url_part = line.split(",", 1)
        name = name.strip()
        url_part = url_part.strip()
        if not name or not url_part:
            continue

        # TVBox supports multiple backup sources separated by '#'
        sub_urls = [u.strip() for u in url_part.split("#") if u.strip()]
        for idx, raw_url in enumerate(sub_urls):
            clean_url = raw_url.split("$", 1)[0].strip()
            if not clean_url.startswith(("http://", "https://", "rtmp://", "rtsp://")):
                continue
            display_name = name if len(sub_urls) == 1 else f"{name} (源{idx + 1})"
            cid = _make_channel_id(display_name, clean_url)
            channels.append(
                ChannelEntry(
                    id=cid,
                    name=display_name,
                    url=clean_url,
                    group=current_group,
                    logo=_default_logo_url(name),
                    tvg_id=name,
                    tvg_name=name,
                    need_translate=is_foreign_channel(name, current_group),
                )
            )
    return channels


def build_proxy_url(base_url: str, channel: ChannelEntry) -> str:
    base = base_url.rstrip("/")
    token = encode_url_token(channel.url)
    ext = "index.m3u8" if channel.is_hls else "stream.ts"
    return f"{base}/live/{channel.id}/{ext}?u={token}&name={quote(channel.name)}"


def export_m3u(
    channels: List[ChannelEntry],
    base_url: str,
    only_foreign: bool = False,
    direct_chinese: bool = False,
    extra_epg_url: str = "",
) -> str:
    base = base_url.rstrip("/")
    url_list = [f"{base}/epg.xml.gz", f"{base}/epg.xml"]
    for u in (extra_epg_url or "https://epg.112114.xyz/pp.xml.gz").split(","):
        u = u.strip()
        if u and u not in url_list and "{name}" not in u:
            url_list.append(u)
    epg_urls = ",".join(url_list)
    out = [f'#EXTM3U x-tvg-url="{epg_urls}" url-tvg="{epg_urls}"']
    for ch in channels:
        if only_foreign and not ch.need_translate:
            continue
        attrs = []
        tvg_id = ch.tvg_id or ch.name
        tvg_name = ch.tvg_name or ch.name
        attrs.append(f'tvg-id="{tvg_id}"')
        attrs.append(f'tvg-name="{tvg_name}"')
        if ch.logo:
            attrs.append(f'tvg-logo="{ch.logo}"')
        if ch.group:
            attrs.append(f'group-title="{ch.group}"')
        attr_str = " " + " ".join(attrs) if attrs else ""
        out.append(f"#EXTINF:-1{attr_str},{ch.name}")
        if direct_chinese and not ch.need_translate:
            out.append(ch.url)
        else:
            out.append(build_proxy_url(base_url, ch))
    return "\n".join(out) + "\n"


def export_txt(
    channels: List[ChannelEntry],
    base_url: str,
    only_foreign: bool = False,
    direct_chinese: bool = False,
) -> str:
    grouped: dict[str, List[ChannelEntry]] = {}
    for ch in channels:
        if only_foreign and not ch.need_translate:
            continue
        grouped.setdefault(ch.group or "国际直播", []).append(ch)

    out: List[str] = []
    for group_name, items in grouped.items():
        out.append(f"{group_name},#genre#")
        for ch in items:
            target_url = ch.url if (direct_chinese and not ch.need_translate) else build_proxy_url(base_url, ch)
            out.append(f"{ch.name},{target_url}")
        out.append("")
    return "\n".join(out).strip() + "\n"
