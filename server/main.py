"""
FastAPI Gateway, Web Console v2 (with SenseVoice + LLM API configuration),
WebSocket Broadcaster, and UDP Auto-Discovery Server for LinguaRelay-TV (声桥 TV 版).
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

import httpx
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, Response, StreamingResponse
from pydantic import BaseModel

from server.ai_pipeline import StreamingSpeechPipeline
from server.playlist_parser import (
    ChannelEntry,
    decode_url_token,
    export_m3u,
    export_txt,
    parse_playlist,
)
from server.stream_proxy import AudioSidecarManager, rewrite_m3u8_manifest

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
PLAYLIST_FILE = DATA_DIR / "playlist_source.txt"
SETTINGS_FILE = DATA_DIR / "settings.json"

DEFAULT_SAMPLE_PLAYLIST = """欧美英语直播,#genre#
BBC News (示例),http://ebsonair.ebs.co.kr/ebs2familypc/familypc1m/playlist.m3u8
CBS News (示例),https://cbsn-us.cbsnstream.cbsnews.com/out/v1/55a8648e8f134e82a470f83d562deeca/master.m3u8
Sky News (示例),https://linear901-oo-hls0-prd-gtm.delivery.skycdp.com/17501/sde-fast-skynews/master.m3u8
"""


def get_lan_ip(peer_ip: str = "192.168.31.1") -> str:
    env_ip = os.environ.get("LINGUARELAY_HOST_IP", "").strip()
    if env_ip:
        return env_ip
    for probe in (peer_ip, "192.168.31.1", "192.168.1.1", "10.10.10.80"):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect((probe, 80))
                ip = s.getsockname()[0]
                if not ip.startswith(("127.", "198.18.", "198.19.", "169.254.")):
                    return ip
        except Exception:
            pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = str(info[4][0])
            if ip.startswith(("192.168.", "10.", "172.")) and not ip.startswith("198.18."):
                return ip
    except Exception:
        pass
    return "127.0.0.1"


class ConnectionHub:
    def __init__(self) -> None:
        self.clients: set[WebSocket] = set()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.clients.add(ws)

    def disconnect(self, ws: WebSocket) -> None:
        self.clients.discard(ws)

    async def broadcast(self, message: dict) -> None:
        if not self.clients:
            return
        raw = json.dumps(message, ensure_ascii=False)
        stale: List[WebSocket] = []
        for ws in list(self.clients):
            try:
                await ws.send_text(raw)
            except Exception:
                stale.append(ws)
        for ws in stale:
            self.clients.discard(ws)


class UDPDiscoveryProtocol(asyncio.DatagramProtocol):
    def __init__(self, http_port: int = 8765) -> None:
        self.http_port = http_port
        self.transport: Optional[asyncio.DatagramTransport] = None

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        self.transport = transport  # type: ignore[assignment]

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        msg = data.decode("utf-8", errors="ignore").strip()
        if msg.startswith("LINGUARELAY_DISCOVER") and self.transport:
            reply = json.dumps(
                {
                    "service": "lingua-relay-tv",
                    "port": self.http_port,
                    "ip": get_lan_ip(addr[0]),
                }
            ).encode("utf-8")
            self.transport.sendto(reply, addr)


hub = ConnectionHub()
_pipeline: Optional[StreamingSpeechPipeline] = None
_sidecar: Optional[AudioSidecarManager] = None
_channels: List[ChannelEntry] = []
_epg_source_url: str = os.environ.get(
    "LINGUARELAY_EPG_URL",
    "https://epg.112114.xyz/pp.xml.gz,https://live.fanmingming.com/e.xml",
).strip()
_epg_last_sync: str = ""


def load_channels_from_disk() -> List[ChannelEntry]:
    global _channels
    if not PLAYLIST_FILE.exists():
        PLAYLIST_FILE.write_text(DEFAULT_SAMPLE_PLAYLIST, encoding="utf-8")
    raw = PLAYLIST_FILE.read_text(encoding="utf-8", errors="ignore")
    _channels = parse_playlist(raw)
    return _channels


def load_settings_into_sidecar(sidecar: AudioSidecarManager) -> None:
    global _epg_source_url
    env_base_url = os.environ.get("LINGUARELAY_LLM_BASE_URL", "https://api.minimaxi.com/v1").strip()
    env_api_key = os.environ.get("LINGUARELAY_LLM_API_KEY", "").strip()
    env_model = os.environ.get("LINGUARELAY_LLM_MODEL", "MiniMax-M3.1-Flash").strip()

    if SETTINGS_FILE.exists():
        try:
            data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            sidecar.subtitle_delay_ms = int(data.get("subtitle_delay_ms", 800))
            sidecar.display_mode = str(data.get("display_mode", "bilingual"))
            sidecar.font_scale = float(data.get("font_scale", 1.0))
            sidecar.auto_sync = bool(data.get("auto_sync", True))
            if data.get("epg_source_url"):
                _epg_source_url = str(data["epg_source_url"]).strip()
            smode = str(data.get("sentence_mode", "full_sentence"))
            if smode in ("full_sentence", "paragraph", "fast_clause"):
                sidecar.sentence_mode = smode
                sidecar.pipeline.sentence_mode = smode

            llm = data.get("llm", {})
            if isinstance(llm, dict):
                sidecar.pipeline.llm_translator.update_config(
                    enabled=bool(llm.get("enabled", True)),
                    base_url=str(llm.get("base_url") or env_base_url),
                    api_key=str(llm.get("api_key") or env_api_key),
                    model=str(llm.get("model") or env_model),
                    source_lang=str(llm.get("source_lang", "auto")),
                )
        except Exception:
            pass
    elif env_api_key or env_base_url or env_model:
        sidecar.pipeline.llm_translator.update_config(
            enabled=True,
            base_url=env_base_url,
            api_key=env_api_key,
            model=env_model,
            source_lang="auto",
        )


def save_settings_from_sidecar(sidecar: AudioSidecarManager) -> None:
    llm_cfg = sidecar.pipeline.llm_translator.config
    data = {
        "subtitle_delay_ms": sidecar.subtitle_delay_ms,
        "display_mode": sidecar.display_mode,
        "font_scale": sidecar.font_scale,
        "sentence_mode": sidecar.pipeline.sentence_mode,
        "auto_sync": sidecar.auto_sync,
        "epg_source_url": _epg_source_url,
        "llm": {
            "enabled": llm_cfg.enabled,
            "base_url": llm_cfg.base_url,
            "api_key": llm_cfg.api_key,
            "model": llm_cfg.model,
            "source_lang": llm_cfg.source_lang,
        },
    }
    SETTINGS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def get_sidecar() -> AudioSidecarManager:
    global _pipeline, _sidecar
    if _sidecar is None:
        _pipeline = StreamingSpeechPipeline()
        _sidecar = AudioSidecarManager(pipeline=_pipeline, broadcast_cb=hub.broadcast)
        load_settings_into_sidecar(_sidecar)
    return _sidecar


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_channels_from_disk()
    sidecar = get_sidecar()
    loop = asyncio.get_running_loop()
    sidecar.attach_loop(loop)

    udp_transport = None
    try:
        udp_port = int(os.environ.get("LINGUARELAY_UDP_PORT", "8766"))
        http_port = int(os.environ.get("LINGUARELAY_PORT", "8765"))
        udp_transport, _ = await loop.create_datagram_endpoint(
            lambda: UDPDiscoveryProtocol(http_port=http_port),
            local_addr=("0.0.0.0", udp_port),
        )
        print(f"[Discovery] UDP auto-discovery listening on 0.0.0.0:{udp_port}")
    except Exception as exc:
        print(f"[Discovery] UDP bind skipped: {exc}")

    yield

    sidecar.stop_current()
    if udp_transport:
        udp_transport.close()


app = FastAPI(title="声桥 TV 版 v2 (LinguaRelay-TV)", lifespan=lifespan)


def _resolve_base_url(request: Request) -> str:
    host = request.headers.get("host")
    if host and not host.startswith(("127.0.0.1", "localhost")):
        return f"{request.url.scheme}://{host}"
    port = request.url.port or int(os.environ.get("LINGUARELAY_PORT", "8765"))
    return f"http://{get_lan_ip()}:{port}"


@app.get("/playlist.m3u")
async def get_playlist_m3u(request: Request, only_foreign: bool = False, direct_chinese: bool = True) -> Response:
    base_url = _resolve_base_url(request)
    body = export_m3u(
        _channels,
        base_url,
        only_foreign=only_foreign,
        direct_chinese=direct_chinese,
        extra_epg_url=_epg_source_url,
    )
    return Response(
        content=body.encode("utf-8"),
        media_type="audio/x-mpegurl; charset=utf-8",
        headers={"Content-Disposition": 'inline; filename="playlist.m3u"'},
    )


@app.get("/playlist_foreign.m3u")
async def get_playlist_foreign_m3u(request: Request) -> Response:
    base_url = _resolve_base_url(request)
    body = export_m3u(_channels, base_url, only_foreign=True, extra_epg_url=_epg_source_url)
    return Response(
        content=body.encode("utf-8"),
        media_type="audio/x-mpegurl; charset=utf-8",
        headers={"Content-Disposition": 'inline; filename="playlist_foreign.m3u"'},
    )


@app.get("/playlist.txt")
async def get_playlist_txt(request: Request, only_foreign: bool = False, direct_chinese: bool = True) -> PlainTextResponse:
    base_url = _resolve_base_url(request)
    body = export_txt(_channels, base_url, only_foreign=only_foreign, direct_chinese=direct_chinese)
    return PlainTextResponse(content=body, media_type="text/plain; charset=utf-8")


@app.get("/playlist_foreign.txt")
async def get_playlist_foreign_txt(request: Request) -> PlainTextResponse:
    base_url = _resolve_base_url(request)
    body = export_txt(_channels, base_url, only_foreign=True)
    return PlainTextResponse(content=body, media_type="text/plain; charset=utf-8")


@app.get("/epg.xml")
async def get_epg_xml() -> Response:
    _ensure_epg_cache()
    epg_file = DATA_DIR / "epg.xml"
    if epg_file.exists():
        return FileResponse(str(epg_file), media_type="application/xml; charset=utf-8")
    fallback_xml = '<?xml version="1.0" encoding="UTF-8"?><tv generator-info-name="SubCast"><channel id="Live"><display-name lang="zh">Live</display-name></channel></tv>'
    return Response(content=fallback_xml.encode("utf-8"), media_type="application/xml; charset=utf-8")


@app.get("/epg.xml.gz")
async def get_epg_xml_gz() -> Response:
    epg_gz = DATA_DIR / "epg.xml.gz"
    if epg_gz.exists():
        return FileResponse(str(epg_gz), media_type="application/gzip", filename="epg.xml.gz")
    raise HTTPException(status_code=404, detail="epg.xml.gz not found")


_epg_cache_mtime: float = 0.0
_epg_alias_map: dict[str, str] = {}
_epg_prog_map: dict[str, list[tuple[str, str, str, str]]] = {}


def _ensure_epg_cache() -> None:
    global _epg_cache_mtime, _epg_alias_map, _epg_prog_map
    epg_file = DATA_DIR / "epg.xml"
    epg_gz = DATA_DIR / "epg.xml.gz"
    if not epg_file.exists() and epg_gz.exists():
        try:
            import gzip as _gzip
            import shutil as _shutil

            with _gzip.open(str(epg_gz), "rb") as f_in, open(str(epg_file), "wb") as f_out:
                _shutil.copyfileobj(f_in, f_out)
        except Exception as exc:
            print(f"[EPG] Decompress warning: {exc}")
    if not epg_file.exists():
        return
    mtime = epg_file.stat().st_mtime
    if mtime == _epg_cache_mtime and _epg_prog_map:
        return
    import re as _re
    import xml.etree.ElementTree as _ET

    def _norm(s: str) -> str:
        return _re.sub(r"[\s\-_HD高清频道台()（）【】\[\]]+", "", s.upper())

    try:
        root = _ET.parse(str(epg_file)).getroot()
        alias_map: dict[str, str] = {}
        prog_map: dict[str, list[tuple[str, str, str, str]]] = {}
        for ch_el in root.findall("channel"):
            cid = (ch_el.get("id") or "").strip()
            if not cid:
                continue
            alias_map[cid.upper()] = cid
            alias_map[_norm(cid)] = cid
            for dn_el in ch_el.findall("display-name"):
                dn = (dn_el.text or "").strip()
                if dn:
                    alias_map[dn.upper()] = cid
                    alias_map[_norm(dn)] = cid
        for pr_el in root.findall("programme"):
            cid = (pr_el.get("channel") or "").strip()
            st = (pr_el.get("start") or "").strip()
            et = (pr_el.get("stop") or "").strip()
            t_el = pr_el.find("title")
            d_el = pr_el.find("desc")
            tit = (t_el.text or "精彩节目").strip() if t_el is not None else "精彩节目"
            dsc = (d_el.text or "").strip() if d_el is not None else ""
            if cid and len(st) >= 12 and len(et) >= 12:
                prog_map.setdefault(cid, []).append((st, et, tit, dsc))
        _epg_alias_map = alias_map
        _epg_prog_map = prog_map
        _epg_cache_mtime = mtime
    except Exception as exc:
        print(f"[EPG] Cache load warning: {exc}")


@app.get("/epg")
@app.get("/api/epg")
async def get_diyp_epg(
    ch: str = Query("", description="Channel name"),
    channel: str = Query("", description="Channel name alias"),
    date: str = Query("", description="Date YYYY-MM-DD"),
) -> dict:
    """DIYP / TVBox (影视仓) compatible JSON EPG endpoint covering all 167 channels in UTC+8."""
    import datetime as _dt
    import re as _re

    _ensure_epg_cache()
    q_name = (ch or channel or "").strip()
    if not date:
        date = _dt.date.today().strftime("%Y-%m-%d")
    d_compact = date.replace("-", "")

    def _norm(s: str) -> str:
        return _re.sub(r"[\s\-_HD高清频道台()（）【】\[\]]+", "", s.upper())

    cid = _epg_alias_map.get(q_name.upper()) or _epg_alias_map.get(_norm(q_name))
    progs = _epg_prog_map.get(cid or "", [])
    day_items = []
    for st, et, tit, dsc in progs:
        if st.startswith(d_compact):
            day_items.append(
                {
                    "start": f"{st[8:10]}:{st[10:12]}",
                    "end": f"{et[8:10]}:{et[10:12]}",
                    "title": tit,
                    "desc": dsc,
                }
            )
    if not day_items and progs:
        # Fallback to latest available day's schedule times if querying a date outside XML window
        avail_dates = sorted({st[:8] for st, _, _, _ in progs})
        ref_d = avail_dates[-1]
        for st, et, tit, dsc in progs:
            if st.startswith(ref_d):
                day_items.append(
                    {
                        "start": f"{st[8:10]}:{st[10:12]}",
                        "end": f"{et[8:10]}:{et[10:12]}",
                        "title": tit,
                        "desc": dsc,
                    }
                )
    return {
        "channel_name": q_name,
        "date": date,
        "epg_data": day_items,
    }


@app.get("/tvbox.json")
async def get_tvbox_json(request: Request) -> dict:
    base_url = _resolve_base_url(request)
    return {
        "lives": [
            {
                "name": "声桥网关全频道(含全套EPG+外语AI字幕)",
                "type": 0,
                "url": f"{base_url}/playlist.m3u",
                "playerType": 2,
                "ua": "okhttp/3.12.11",
                "epg": f"{base_url}/epg?ch={{name}}&date={{date}}",
                "logo": "https://live.fanmingming.com/tv/{name}.png",
            },
            {
                "name": "外语实时翻译专区(30个国际/英语台)",
                "type": 0,
                "url": f"{base_url}/playlist_foreign.m3u",
                "playerType": 2,
                "ua": "okhttp/3.12.11",
                "epg": f"{base_url}/epg?ch={{name}}&date={{date}}",
                "logo": "https://live.fanmingming.com/tv/{name}.png",
            },
        ]
    }


@app.get("/live/{channel_id}/index.m3u8")
async def proxy_hls_manifest(
    channel_id: str,
    request: Request,
    u: str = Query(..., description="Base64URL encoded upstream URL"),
    name: str = Query("Live Channel", description="Channel display name"),
) -> Response:
    try:
        upstream_url = decode_url_token(u)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid stream token: {exc}") from exc

    root_url = upstream_url
    need_translate = True
    for ch in _channels:
        if ch.id == channel_id:
            root_url = ch.url
            name = ch.name or name
            need_translate = ch.need_translate
            break

    sidecar = get_sidecar()
    if need_translate:
        sidecar.touch_channel(channel_id=channel_id, channel_name=name, upstream_url=root_url)
    elif sidecar.active_channel_id and not sidecar.manual_pinned:
        sidecar.stop_current()
        await hub.broadcast({"type": "clear"})

    headers = {
        "User-Agent": request.headers.get(
            "user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) LinguaRelayTV/2.0"
        )
    }
    async with httpx.AsyncClient(follow_redirects=True, timeout=15.0, verify=False) as client:
        resp = await client.get(upstream_url, headers=headers)
        if resp.status_code >= 400:
            raise HTTPException(status_code=resp.status_code, detail="Upstream M3U8 error")
        final_url = str(resp.url)
        text = resp.text

    if need_translate:
        sidecar.update_hls_manifest_metrics(text)

    base_url = _resolve_base_url(request)
    rewritten = rewrite_m3u8_manifest(
        manifest_text=text,
        final_upstream_url=final_url,
        base_proxy_url=base_url,
        channel_id=channel_id,
        channel_name=name,
    )
    return Response(
        content=rewritten.encode("utf-8"),
        media_type="application/vnd.apple.mpegurl",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@app.get("/live/{channel_id}/stream.ts")
async def proxy_direct_stream(
    channel_id: str,
    request: Request,
    u: str = Query(..., description="Base64URL encoded upstream URL"),
    name: str = Query("Live Channel", description="Channel display name"),
) -> StreamingResponse:
    try:
        upstream_url = decode_url_token(u)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid stream token: {exc}") from exc

    sidecar = get_sidecar()
    sidecar.touch_channel(channel_id=channel_id, channel_name=name, upstream_url=upstream_url)

    async def _stream_generator():
        client = httpx.AsyncClient(follow_redirects=True, timeout=None, verify=False)
        try:
            async with client.stream("GET", upstream_url) as resp:
                async for chunk in resp.aiter_bytes(chunk_size=32768):
                    sidecar.touch_channel(
                        channel_id=channel_id, channel_name=name, upstream_url=upstream_url
                    )
                    yield chunk
        finally:
            await client.aclose()

    return StreamingResponse(_stream_generator(), media_type="video/mp2t")


@app.websocket("/ws/subtitles")
async def websocket_subtitles(ws: WebSocket) -> None:
    await hub.connect(ws)
    sidecar = get_sidecar()
    status = sidecar.get_status()
    await ws.send_text(
        json.dumps(
            {
                "type": "connected",
                "status": asdict(status),
                "recent": sidecar.recent_subtitles[-5:],
            },
            ensure_ascii=False,
        )
    )
    try:
        while True:
            msg_text = await ws.receive_text()
            try:
                data = json.loads(msg_text)
                action = data.get("action")
                if action == "update_settings":
                    if "subtitle_delay_ms" in data:
                        sidecar.subtitle_delay_ms = max(0, min(8000, int(data["subtitle_delay_ms"])))
                    if "display_mode" in data:
                        sidecar.display_mode = str(data["display_mode"])
                    if "font_scale" in data:
                        sidecar.font_scale = float(data["font_scale"])
                    if data.get("sentence_mode") in ("full_sentence", "paragraph", "fast_clause"):
                        sidecar.sentence_mode = str(data["sentence_mode"])
                        sidecar.pipeline.sentence_mode = str(data["sentence_mode"])
                    save_settings_from_sidecar(sidecar)
                    await hub.broadcast({"type": "settings", "status": asdict(sidecar.get_status())})
                elif action == "ping":
                    await ws.send_text(json.dumps({"type": "pong", "ts": time.time()}))
            except Exception:
                pass
    except WebSocketDisconnect:
        hub.disconnect(ws)
    except Exception:
        hub.disconnect(ws)


class PlaylistUpdatePayload(BaseModel):
    content: Optional[str] = None
    url: Optional[str] = None


class SettingsUpdatePayload(BaseModel):
    subtitle_delay_ms: Optional[int] = None
    display_mode: Optional[str] = None
    font_scale: Optional[float] = None
    sentence_mode: Optional[str] = None
    auto_sync: Optional[bool] = None


class LLMConfigPayload(BaseModel):
    enabled: bool = True
    base_url: str = "https://api.minimaxi.com/v1"
    api_key: str = ""
    model: str = "MiniMax-M3.1-Flash"
    source_lang: str = "auto"


class PinChannelPayload(BaseModel):
    channel_id: Optional[str] = "manual"
    name: str = "Manual Stream"
    url: str


class TestSubtitlePayload(BaseModel):
    en: str = "Good evening, breaking news from our live broadcast studio in London."
    zh: str = ""


class EpgSourcePayload(BaseModel):
    url: str


async def sync_remote_epg(url_str: str) -> dict:
    """Download remote XMLTV (.xml / .xml.gz) or validate DIYP EPG URL, merging into local data/epg.xml."""
    global _epg_source_url, _epg_last_sync, _epg_cache_mtime
    import datetime as _dt
    import gzip as _gzip
    import xml.etree.ElementTree as _ET

    clean_urls = [u.strip() for u in (url_str or "").split(",") if u.strip()]
    if not clean_urls:
        raise HTTPException(status_code=400, detail="EPG 地址不能为空")

    _epg_source_url = ",".join(clean_urls)
    save_settings_from_sidecar(get_sidecar())

    xml_urls = [u for u in clean_urls if "{name}" not in u and "?ch=" not in u]
    diyp_urls = [u for u in clean_urls if u not in xml_urls]

    fetched_channels = 0
    fetched_progs = 0
    errors: list[str] = []

    if xml_urls:
        epg_file = DATA_DIR / "epg.xml"
        epg_gz = DATA_DIR / "epg.xml.gz"
        base_root = _ET.Element("tv", {"generator-info-name": "SubCast-EPG-Aggregator"})
        seen_channels: dict[str, _ET.Element] = {}
        progs_by_ch: dict[str, list[_ET.Element]] = {}

        # Load existing local EPG first so channels not in remote source keep their schedules
        if epg_file.exists():
            try:
                old_root = _ET.parse(str(epg_file)).getroot()
                for ch_el in old_root.findall("channel"):
                    cid = (ch_el.get("id") or "").strip()
                    if cid:
                        seen_channels[cid] = ch_el
                for pr_el in old_root.findall("programme"):
                    cid = (pr_el.get("channel") or "").strip()
                    if cid:
                        progs_by_ch.setdefault(cid, []).append(pr_el)
            except Exception:
                pass

        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=35.0,
            verify=False,
            headers={"User-Agent": "Mozilla/5.0 SubCast/2.4"},
        ) as client:
            for u in xml_urls:
                try:
                    resp = await client.get(u)
                    resp.raise_for_status()
                    raw_bytes = resp.content
                    if raw_bytes[:2] == b"\x1f\x8b" or u.lower().split("?")[0].endswith(".gz"):
                        raw_bytes = _gzip.decompress(raw_bytes)
                    xml_str = raw_bytes.decode("utf-8", errors="ignore")
                    if "epg.pw" in u.lower() and any(tag in u.upper() for tag in ("_CN", "_HK", "_TW", "_SG")):
                        xml_str = xml_str.replace(" +0000", " +0800")
                    remote_root = _ET.fromstring(xml_str)
                    remote_ch_ids: set[str] = set()
                    for ch_el in remote_root.findall("channel"):
                        cid = (ch_el.get("id") or "").strip()
                        if cid:
                            seen_channels[cid] = ch_el
                            remote_ch_ids.add(cid)
                    # Replace programmes for channels updated by this remote source
                    for cid in remote_ch_ids:
                        progs_by_ch[cid] = []
                    for pr_el in remote_root.findall("programme"):
                        cid = (pr_el.get("channel") or "").strip()
                        if cid:
                            progs_by_ch.setdefault(cid, []).append(pr_el)
                            fetched_progs += 1
                    fetched_channels += len(remote_ch_ids)
                except Exception as exc:
                    errors.append(f"{u}: {exc}")

        if fetched_channels > 0 or fetched_progs > 0:
            for cid in sorted(seen_channels.keys()):
                base_root.append(seen_channels[cid])
            for cid in sorted(progs_by_ch.keys()):
                for pr_el in progs_by_ch[cid]:
                    base_root.append(pr_el)
            xml_bytes = _ET.tostring(base_root, encoding="utf-8", xml_declaration=True)
            epg_file.write_bytes(xml_bytes)
            with _gzip.open(str(epg_gz), "wb") as gz_out:
                gz_out.write(xml_bytes)
            _epg_cache_mtime = 0.0
            _ensure_epg_cache()

    if diyp_urls and not xml_urls:
        _ensure_epg_cache()

    if errors and fetched_channels == 0 and not diyp_urls:
        return {
            "ok": False,
            "error": "; ".join(errors),
            "epg_source_url": _epg_source_url,
        }

    _epg_last_sync = _dt.datetime.now().strftime("%H:%M:%S")
    _ensure_epg_cache()
    total_progs = sum(len(v) for v in _epg_prog_map.values())
    return {
        "ok": True,
        "epg_source_url": _epg_source_url,
        "channel_count": len(_epg_prog_map),
        "programme_count": total_progs,
        "synced_at": _epg_last_sync,
        "warnings": errors,
    }


@app.post("/api/epg-source")
async def api_update_epg_source(payload: EpgSourcePayload) -> dict:
    return await sync_remote_epg(payload.url)


@app.get("/api/status")
async def api_status(request: Request) -> dict:
    sidecar = get_sidecar()
    base_url = _resolve_base_url(request)
    llm_cfg = sidecar.pipeline.llm_translator.config
    _ensure_epg_cache()
    return {
        "lan_ip": get_lan_ip(),
        "base_url": base_url,
        "m3u_url": f"{base_url}/playlist.m3u",
        "txt_url": f"{base_url}/playlist.txt",
        "epg_xml_url": f"{base_url}/epg.xml",
        "epg_diyp_url": f"{base_url}/epg?ch={{name}}&date={{date}}",
        "epg_source_url": _epg_source_url,
        "epg_channel_count": len(_epg_prog_map),
        "epg_last_sync": _epg_last_sync,
        "ws_url": f"{base_url.replace('http://', 'ws://').replace('https://', 'wss://')}/ws/subtitles",
        "tv_clients_connected": len(hub.clients),
        "sidecar": asdict(sidecar.get_status()),
        "llm_config": {
            "enabled": llm_cfg.enabled,
            "base_url": llm_cfg.base_url,
            "api_key_masked": (
                f"{llm_cfg.api_key[:6]}...{llm_cfg.api_key[-4:]}"
                if len(llm_cfg.api_key) > 10
                else ("已设置" if llm_cfg.api_key else "")
            ),
            "has_api_key": bool(llm_cfg.api_key),
            "model": llm_cfg.model,
            "source_lang": llm_cfg.source_lang,
        },
        "channel_count": len(_channels),
        "channels": [asdict(ch) for ch in _channels],
        "recent_subtitles": sidecar.recent_subtitles[-20:],
    }


@app.post("/api/llm-config")
async def api_update_llm_config(payload: LLMConfigPayload) -> dict:
    sidecar = get_sidecar()
    # Keep existing key if user didn't re-enter and sent masked placeholder
    new_key = payload.api_key.strip()
    if "..." in new_key and sidecar.pipeline.llm_translator.config.api_key:
        new_key = sidecar.pipeline.llm_translator.config.api_key

    sidecar.pipeline.llm_translator.update_config(
        enabled=payload.enabled,
        base_url=payload.base_url,
        api_key=new_key,
        model=payload.model,
        source_lang=payload.source_lang,
    )
    save_settings_from_sidecar(sidecar)

    # Test translation immediately so user gets instant feedback
    t0 = time.perf_counter()
    sample_en = "Breaking news: central bank officials are meeting in London to discuss global interest rates."
    zh_out = sidecar.pipeline.llm_translator.translate(sample_en, channel_name="BBC World News", is_final=False)
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)

    return {
        "ok": True,
        "engine_used": sidecar.pipeline.llm_translator.last_used_engine,
        "error": sidecar.pipeline.llm_translator.last_error,
        "sample_en": sample_en,
        "sample_zh": zh_out,
        "latency_ms": elapsed_ms,
    }


@app.get("/api/playlist/raw")
async def api_get_raw_playlist() -> PlainTextResponse:
    if not PLAYLIST_FILE.exists():
        return PlainTextResponse("")
    return PlainTextResponse(PLAYLIST_FILE.read_text(encoding="utf-8", errors="ignore"))


@app.post("/api/playlist")
async def api_update_playlist(payload: PlaylistUpdatePayload) -> dict:
    raw_text = payload.content or ""
    if payload.url and not raw_text.strip():
        async with httpx.AsyncClient(follow_redirects=True, timeout=20.0, verify=False) as client:
            resp = await client.get(payload.url)
            resp.raise_for_status()
            raw_text = resp.text

    if not raw_text.strip():
        raise HTTPException(status_code=400, detail="播放列表内容不能为空")

    PLAYLIST_FILE.write_text(raw_text, encoding="utf-8")
    channels = load_channels_from_disk()
    return {"ok": True, "channel_count": len(channels)}


@app.post("/api/settings")
async def api_update_settings(payload: SettingsUpdatePayload) -> dict:
    sidecar = get_sidecar()
    if payload.subtitle_delay_ms is not None:
        sidecar.subtitle_delay_ms = max(0, min(8000, payload.subtitle_delay_ms))
    if payload.display_mode in ("bilingual", "zh_only"):
        sidecar.display_mode = payload.display_mode
    if payload.font_scale is not None:
        sidecar.font_scale = max(0.6, min(2.0, payload.font_scale))
    if payload.sentence_mode in ("full_sentence", "paragraph", "fast_clause"):
        sidecar.sentence_mode = payload.sentence_mode
        sidecar.pipeline.sentence_mode = payload.sentence_mode
    if payload.auto_sync is not None:
        sidecar.auto_sync = payload.auto_sync
    save_settings_from_sidecar(sidecar)
    status_dict = asdict(sidecar.get_status())
    await hub.broadcast({"type": "settings", "status": status_dict})
    return {"ok": True, "sidecar": status_dict}


@app.post("/api/pin")
async def api_pin_channel(payload: PinChannelPayload) -> dict:
    sidecar = get_sidecar()
    sidecar.touch_channel(
        channel_id=payload.channel_id or "manual",
        channel_name=payload.name,
        upstream_url=payload.url,
        manual_pin=True,
    )
    return {"ok": True, "sidecar": asdict(sidecar.get_status())}


@app.post("/api/stop")
async def api_stop_sidecar() -> dict:
    sidecar = get_sidecar()
    sidecar.stop_current()
    await hub.broadcast({"type": "clear"})
    return {"ok": True, "sidecar": asdict(sidecar.get_status())}


@app.post("/api/test-subtitle")
async def api_send_test_subtitle(payload: TestSubtitlePayload) -> dict:
    sidecar = get_sidecar()
    ev = sidecar.pipeline.process_text_direct(payload.en, is_final=True)
    msg = {
        "type": "subtitle",
        "channel": "测试字幕",
        "en": ev.en if ev else payload.en,
        "zh": ev.zh if (ev and ev.zh) else payload.zh,
        "is_final": True,
        "timestamp": time.time(),
        "latency_ms": ev.latency_ms if ev else 0.0,
        "engine_info": ev.engine_info if ev else "",
        "display_mode": sidecar.display_mode,
        "font_scale": sidecar.font_scale,
    }
    await hub.broadcast(msg)
    return {"ok": True, "subtitle": msg}


WEB_CONSOLE_HTML = """<!DOCTYPE html>
<html lang="zh-CN" data-radius="soft" data-buttons="solid">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>声幕 SubCast — 实时直播同传字幕网关</title>
<link rel="icon" type="image/svg+xml" href="/logo.svg">
<link rel="apple-touch-icon" href="/logo.png">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Geist:wght@400..600&family=Geist+Mono:wght@400..500&display=swap" rel="stylesheet">
<style>
  /* ----------------------------------------------------------------
     tasteskill 2.5 system tokens
     Palette: steel-orange (Day) / graphite-safety-orange (Evening)
     Type: Geist + Geist Mono (display 500, tracking -0.035em)
     Rule: 92% neutral, 1 accent (<3% screen), zero violet, max 2 boxes deep
     ---------------------------------------------------------------- */
  :root {
    --p-bg: #eeeeeb;
    --p-surface: #e3e3de;
    --p-card: #f6f6f3;
    --p-ink: #1c1d1f;
    --p-muted: #5c5e62;
    --p-accent: #d94814;
    --p-accent-ink: #ffffff;
    --p-action: #d94814;
    --p-action-ink: #ffffff;
    --p-inv-bg: #161719;
    --p-inv-surface: #222427;
    --p-inv-ink: #eeeeeb;
    --p-inv-muted: #96989c;

    --font-display: "Geist", -apple-system, BlinkMacSystemFont, "PingFang SC", "Microsoft YaHei", sans-serif;
    --font-text: "Geist", -apple-system, BlinkMacSystemFont, "PingFang SC", "Microsoft YaHei", sans-serif;
    --font-mono: "Geist Mono", ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace;
    --display-weight: 500;
    --display-tracking: -0.03em;

    --s-1: 4px;  --s-2: 8px;  --s-3: 12px; --s-4: 16px;
    --s-5: 24px; --s-6: 32px; --s-7: 48px; --s-8: 64px;

    --r-sm: 6px; --r-md: 10px; --r-lg: 14px; --r-btn: 6px;
    --btn-h: 40px;
    --btn-h-sm: 32px;
    --ease: cubic-bezier(.2, .8, .2, 1);
    --dur: 180ms;
  }

  html[data-tone="dark"] {
    --p-bg: #18191b;
    --p-surface: #222427;
    --p-card: #1e2023;
    --p-ink: #eff0f1;
    --p-muted: #94979c;
    --p-accent: #ff4a1f;
    --p-accent-ink: #18191b;
    --p-action: #ff4a1f;
    --p-action-ink: #18191b;
    --p-inv-bg: #111214;
    --p-inv-surface: #1b1c1f;
    --p-inv-ink: #eff0f1;
    --p-inv-muted: #888b90;
  }

  :root, .ground-base {
    --bg: var(--p-bg);
    --surface: var(--p-surface);
    --card: var(--p-card);
    --ink: var(--p-ink);
    --muted: var(--p-muted);
    --accent: var(--p-accent);
    --accent-ink: var(--p-accent-ink);
    --action: var(--p-action);
    --action-ink: var(--p-action-ink);
    --line: color-mix(in oklab, var(--ink) 12%, transparent);
    --line-strong: color-mix(in oklab, var(--ink) 24%, transparent);
  }

  .ground-inverse {
    --bg: var(--p-inv-bg);
    --surface: var(--p-inv-surface);
    --ink: var(--p-inv-ink);
    --muted: var(--p-inv-muted);
    --line: color-mix(in oklab, var(--ink) 14%, transparent);
    --line-strong: color-mix(in oklab, var(--ink) 26%, transparent);
    background: var(--bg);
    color: var(--ink);
  }

  *, *::before, *::after { box-sizing: border-box; }
  html, body { margin: 0; }
  body {
    background: var(--bg);
    color: var(--ink);
    font: 400 14px/1.55 var(--font-text);
    -webkit-font-smoothing: antialiased;
    text-rendering: optimizeLegibility;
    padding-top: 64px;
    padding-bottom: var(--s-8);
  }

  h1, h2, h3, p, ul, ol { margin: 0; }
  ul, ol { padding: 0; list-style: none; }
  button, input, textarea { font: inherit; color: inherit; }

  .muted { color: var(--muted); font-weight: 400; }
  .mono { font-family: var(--font-mono); font-variant-numeric: tabular-nums; }

  :focus-visible {
    outline: 2px solid var(--action);
    outline-offset: 2px;
  }

  .wrap {
    width: min(100% - 48px, 1240px);
    margin-inline: auto;
  }

  /* ── Fixed Opaque Header (.nav from system.css) ── */
  .nav {
    position: fixed;
    inset: 0 0 auto 0;
    z-index: 50;
    height: 64px;
    background: var(--bg);
    color: var(--ink);
    border-bottom: 1px solid var(--line);
  }
  .nav__bar {
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--s-5);
  }
  .nav__logo {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    font: 500 19px/1 var(--font-display);
    letter-spacing: -0.025em;
    color: var(--ink);
    text-decoration: none;
  }
  /* Dealt mark: stack (system.css line 657) */
  .mark {
    display: inline-block;
    flex: none;
    width: 20px;
    height: 20px;
    background: var(--accent);
    -webkit-mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Crect x='2' y='2.5' width='20' height='5.5' rx='2.75'/%3E%3Crect x='2' y='9.25' width='14' height='5.5' rx='2.75'/%3E%3Crect x='2' y='16' width='8' height='5.5' rx='2.75'/%3E%3C/svg%3E") center / contain no-repeat;
            mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Crect x='2' y='2.5' width='20' height='5.5' rx='2.75'/%3E%3Crect x='2' y='9.25' width='14' height='5.5' rx='2.75'/%3E%3Crect x='2' y='16' width='8' height='5.5' rx='2.75'/%3E%3C/svg%3E") center / contain no-repeat;
  }
  .nav__telemetry {
    display: flex;
    align-items: center;
    gap: var(--s-5);
    flex-wrap: wrap;
  }
  .nav__stat {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-size: 13px;
  }
  .nav__stat-label {
    color: var(--muted);
    font-weight: 500;
  }
  .nav__stat-val {
    font-family: var(--font-mono);
    font-variant-numeric: tabular-nums;
    font-weight: 500;
    color: var(--ink);
  }
  .status-dot {
    width: 7px;
    height: 7px;
    border-radius: 999px;
    background: var(--line-strong);
    flex-shrink: 0;
  }
  .status-dot.is-live {
    background: var(--accent);
  }
  .nav__actions {
    display: flex;
    align-items: center;
    gap: var(--s-2);
  }

  /* ── Layout Grid (12-column system.css) ── */
  .main-stack {
    display: grid;
    gap: var(--s-5);
    padding-top: var(--s-5);
  }
  .grid-12 {
    display: grid;
    grid-template-columns: repeat(12, minmax(0, 1fr));
    gap: var(--s-5);
  }
  .span-6 { grid-column: span 6; }
  .span-12 { grid-column: span 12; }
  @media (max-width: 960px) {
    .span-6 { grid-column: span 12; }
    .nav__telemetry { display: none; }
  }

  /* ── Cards (One recipe, flat 1px hairline, zero shadow) ── */
  .card {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: var(--r-lg);
    padding: var(--s-5);
    display: flex;
    flex-direction: column;
    gap: var(--s-4);
  }
  .section-head--split {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: var(--s-4);
    flex-wrap: wrap;
    padding-bottom: var(--s-3);
    border-bottom: 1px solid var(--line);
  }
  h2 {
    font-family: var(--font-display);
    font-weight: var(--display-weight);
    font-size: 18px;
    line-height: 1.25;
    letter-spacing: var(--display-tracking);
    display: flex;
    align-items: center;
    gap: 10px;
  }
  h2 .muted {
    font-size: 14px;
    letter-spacing: 0;
    margin-left: 4px;
  }
  .step-num {
    display: inline-grid;
    place-items: center;
    width: 26px;
    height: 26px;
    border-radius: 999px;
    border: 1px solid var(--line-strong);
    font: 500 12px/1 var(--font-mono);
    font-variant-numeric: tabular-nums;
    color: var(--ink);
    flex: none;
  }

  /* ── Buttons (One pair: .btn--primary & .btn--secondary) ── */
  .btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: var(--s-2);
    height: var(--btn-h);
    padding: 0 var(--s-4);
    border-radius: var(--r-btn);
    border: 1px solid transparent;
    font: 500 13.5px/1 var(--font-text);
    white-space: nowrap;
    cursor: pointer;
    text-decoration: none;
    user-select: none;
    transition: background-color var(--dur) var(--ease), color var(--dur) var(--ease), border-color var(--dur) var(--ease), transform var(--dur) var(--ease);
  }
  .btn:active {
    transform: translateY(1px) scale(.985);
  }
  .btn--primary {
    background: var(--action);
    color: var(--action-ink);
  }
  .btn--primary:hover {
    background: color-mix(in oklab, var(--action) 86%, var(--ink));
  }
  .btn--secondary {
    background: var(--bg);
    color: var(--ink);
    border-color: var(--line-strong);
  }
  .btn--secondary:hover {
    background: color-mix(in oklab, var(--bg) 92%, var(--ink));
    border-color: color-mix(in oklab, var(--ink) 38%, transparent);
  }
  .btn--sm {
    height: var(--btn-h-sm);
    padding: 0 var(--s-3);
    font-size: 13px;
  }
  .m-arrow {
    flex: none;
    width: 11px;
    height: 11px;
    fill: none;
    stroke: currentColor;
    stroke-width: 1.6;
    stroke-linecap: round;
    stroke-linejoin: round;
    transition: transform var(--dur) var(--ease);
  }
  .btn:hover .m-arrow {
    transform: translateX(2.5px);
  }

  /* ── Tabs / Segmented Controls (.tabs & .tab from system.css) ── */
  .tabs {
    display: inline-flex;
    flex-wrap: wrap;
    gap: 4px;
  }
  .tab {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    height: var(--btn-h-sm);
    padding: 0 12px;
    border-radius: var(--r-btn);
    border: 1px solid var(--line);
    background: transparent;
    color: var(--muted);
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    transition: background-color var(--dur) var(--ease), color var(--dur) var(--ease), border-color var(--dur) var(--ease);
  }
  .tab:hover {
    color: var(--ink);
    border-color: var(--line-strong);
  }
  .tab[aria-selected="true"] {
    background: var(--ink);
    color: var(--bg);
    border-color: var(--ink);
  }
  .brand-icon {
    width: 14px;
    height: 14px;
    flex-shrink: 0;
    display: inline-block;
  }

  /* ── Monitor Frame (.ground-inverse) & Live Audio Pulse Meter ── */
  .monitor-frame {
    border-radius: var(--r-md);
    border: 1px solid var(--line);
    overflow: clip;
  }
  .monitor-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 18px;
    border-bottom: 1px solid var(--line);
    font-size: 12.5px;
    flex-wrap: wrap;
    gap: var(--s-3);
  }
  .monitor-ch {
    display: inline-flex;
    align-items: center;
    gap: 9px;
    font-weight: 500;
    color: var(--ink);
  }
  .vu-meter {
    display: inline-flex;
    align-items: flex-end;
    gap: 2px;
    height: 13px;
    padding-bottom: 1px;
    margin-left: 2px;
  }
  .vu-meter i {
    display: block;
    width: 2px;
    height: 3px;
    border-radius: 1px;
    background: var(--muted);
    opacity: 0.45;
    transition: height 160ms var(--ease), background-color 160ms var(--ease), opacity 160ms var(--ease);
  }
  .vu-meter.is-live i {
    background: var(--accent);
    opacity: 1;
    animation: vu-pulse 0.64s ease-in-out infinite alternate;
  }
  .vu-meter.is-live i:nth-child(1) { animation-duration: 0.52s; animation-delay: 0.00s; }
  .vu-meter.is-live i:nth-child(2) { animation-duration: 0.68s; animation-delay: 0.14s; }
  .vu-meter.is-live i:nth-child(3) { animation-duration: 0.44s; animation-delay: 0.07s; }
  .vu-meter.is-live i:nth-child(4) { animation-duration: 0.61s; animation-delay: 0.21s; }
  .vu-meter.is-live i:nth-child(5) { animation-duration: 0.55s; animation-delay: 0.11s; }
  @keyframes vu-pulse {
    0%   { height: 3px; }
    45%  { height: 8px; }
    100% { height: 13px; }
  }
  @media (prefers-reduced-motion: reduce) {
    .vu-meter.is-live i { animation: none; height: 9px; }
  }
  .ch-logo {
    width: 22px;
    height: 16px;
    object-fit: contain;
    flex-shrink: 0;
    border-radius: 2px;
  }
  .monitor-metrics {
    display: flex;
    gap: var(--s-4);
    font-family: var(--font-mono);
    font-variant-numeric: tabular-nums;
    font-size: 12px;
    color: var(--muted);
  }
  .monitor-stage {
    min-height: 148px;
    padding: 28px 24px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
    gap: 6px;
  }
  .sub-en {
    font-size: 15.5px;
    font-weight: 400;
    color: #d4d5d8;
    line-height: 1.45;
    max-width: 68ch;
  }
  .sub-zh {
    font-size: 23px;
    font-weight: 600;
    color: #f5d90a;
    line-height: 1.4;
    max-width: 52ch;
    letter-spacing: -0.01em;
  }
  .sub-zh.placeholder {
    font-size: 14px;
    font-weight: 400;
    color: var(--muted);
  }

  /* ── Controls Row ── */
  .ctrl-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: var(--s-4);
    padding-top: var(--s-2);
  }
  .ctrl-group {
    display: flex;
    align-items: center;
    gap: var(--s-4);
    flex-wrap: wrap;
  }
  .ctrl-item {
    display: inline-flex;
    align-items: center;
    gap: var(--s-2);
  }
  .ctrl-label {
    font-size: 13px;
    font-weight: 500;
    color: var(--muted);
  }

  /* Toggle Switch */
  .toggle {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    cursor: pointer;
    font-size: 13px;
    font-weight: 500;
    user-select: none;
  }
  .toggle input { display: none; }
  .toggle__track {
    width: 34px;
    height: 18px;
    border-radius: 999px;
    background: var(--surface);
    border: 1px solid var(--line-strong);
    position: relative;
    transition: background-color var(--dur) var(--ease), border-color var(--dur) var(--ease);
  }
  .toggle__thumb {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: var(--ink);
    position: absolute;
    top: 2px;
    left: 2px;
    transition: transform var(--dur) var(--ease), background-color var(--dur) var(--ease);
  }
  .toggle input:checked + .toggle__track {
    background: var(--ink);
    border-color: var(--ink);
  }
  .toggle input:checked + .toggle__track .toggle__thumb {
    background: var(--bg);
    transform: translateX(16px);
  }

  /* Range Slider */
  input[type="range"] {
    -webkit-appearance: none;
    appearance: none;
    width: 108px;
    height: 3px;
    background: var(--line-strong);
    border-radius: 2px;
    outline: none;
  }
  input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 14px;
    height: 14px;
    border-radius: 50%;
    background: var(--ink);
    border: 2px solid var(--bg);
    cursor: pointer;
  }

  /* ── Hairline Lists (.rows from system.css line 1226 — no box-in-a-box) ── */
  .rows {
    border-top: 0;
  }
  .rows > li {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    align-items: center;
    gap: var(--s-4);
    padding-block: 10px;
    border-bottom: 1px solid var(--line);
  }
  .rows > li:first-child {
    padding-top: 2px;
  }
  .rows > li:last-child {
    border-bottom: 0;
    padding-bottom: 2px;
  }
  .rows__name {
    font-weight: 500;
    color: var(--ink);
    display: flex;
    align-items: center;
    gap: 8px;
    min-width: 0;
  }
  .rows__sub {
    font-family: var(--font-mono);
    font-size: 12.5px;
    color: var(--muted);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .ch-scroll {
    max-height: 290px;
    overflow-y: auto;
    padding-right: 4px;
  }

  /* ── Forms (.field from system.css line 1240) ── */
  .api-grid {
    display: grid;
    grid-template-columns: 1.2fr 1.3fr 0.9fr auto;
    gap: var(--s-3);
    align-items: end;
  }
  @media (max-width: 900px) { .api-grid { grid-template-columns: 1fr; } }
  .field {
    display: grid;
    gap: 6px;
    font-size: 13px;
    font-weight: 500;
    color: var(--ink);
  }
  .field label {
    color: var(--muted);
    font-size: 12.5px;
  }
  .field input, textarea {
    width: 100%;
    height: var(--btn-h);
    padding: 0 12px;
    border: 1px solid var(--line-strong);
    border-radius: var(--r-sm);
    background: var(--bg);
    color: var(--ink);
    font: 400 13.5px/1.4 var(--font-text);
    transition: border-color var(--dur) var(--ease);
  }
  .field input:focus, textarea:focus {
    outline: none;
    border-color: var(--ink);
  }
  textarea {
    height: auto;
    min-height: 332px;
    padding: 12px;
    font-family: var(--font-mono);
    font-size: 12.5px;
    line-height: 1.6;
    resize: vertical;
  }
  .search-input {
    height: var(--btn-h-sm);
    width: 180px;
    padding: 0 10px;
    border: 1px solid var(--line-strong);
    border-radius: var(--r-sm);
    background: var(--bg);
    color: var(--ink);
    font-size: 13px;
  }
  .search-input:focus {
    outline: none;
    border-color: var(--ink);
  }
</style>
</head>
<body>

<!-- Official Brand Vector Marks (Rule 7: Real full-colour product icons) -->
<svg width="0" height="0" style="position:absolute; width:0; height:0; overflow:hidden; pointer-events:none;" aria-hidden="true">
  <defs>
    <linearGradient id="mm-grad" x1="0" y1="12" x2="24" y2="12" gradientUnits="userSpaceOnUse">
      <stop offset="0%" stop-color="#E2167E"/>
      <stop offset="50%" stop-color="#F03B5D"/>
      <stop offset="100%" stop-color="#FE603C"/>
    </linearGradient>
    <symbol id="ico-minimax" viewBox="0 0 24 24">
      <path d="M16.278 2c1.156 0 2.093.927 2.093 2.07v12.501a.74.74 0 00.744.709.74.74 0 00.743-.709V9.099a2.06 2.06 0 012.071-2.049A2.06 2.06 0 0124 9.1v6.561a.649.649 0 01-.652.645.649.649 0 01-.653-.645V9.1a.762.762 0 00-.766-.758.762.762 0 00-.766.758v7.472a2.037 2.037 0 01-2.048 2.026 2.037 2.037 0 01-2.048-2.026v-12.5a.785.785 0 00-.788-.753.785.785 0 00-.789.752l-.001 15.904A2.037 2.037 0 0113.441 22a2.037 2.037 0 01-2.048-2.026V18.04c0-.356.292-.645.652-.645.36 0 .652.289.652.645v1.934c0 .263.142.506.372.638.23.131.514.131.744 0a.734.734 0 00.372-.638V4.07c0-1.143.937-2.07 2.093-2.07zm-5.674 0c1.156 0 2.093.927 2.093 2.07v11.523a.648.648 0 01-.652.645.648.648 0 01-.652-.645V4.07a.785.785 0 00-.789-.78.785.785 0 00-.789.78v14.013a2.06 2.06 0 01-2.07 2.048 2.06 2.06 0 01-2.071-2.048V9.1a.762.762 0 00-.766-.758.762.762 0 00-.766.758v3.8a2.06 2.06 0 01-2.071 2.049A2.06 2.06 0 010 12.9v-1.378c0-.357.292-.646.652-.646.36 0 .653.29.653.646V12.9c0 .418.343.757.766.757s.766-.339.766-.757V9.099a2.06 2.06 0 012.07-2.048 2.06 2.06 0 012.071 2.048v8.984c0 .419.343.758.767.758.423 0 .766-.339.766-.758V4.07c0-1.143.937-2.07 2.093-2.07z" fill="#F03B5D" style="fill:url(#mm-grad)"/>
    </symbol>
    <symbol id="ico-deepseek" viewBox="0 0 24 24">
      <path d="M23.748 4.482c-.254-.124-.364.113-.512.234-.051.039-.094.09-.137.136-.372.397-.806.657-1.373.626-.829-.046-1.537.214-2.163.848-.133-.782-.575-1.248-1.247-1.548-.352-.156-.708-.311-.955-.65-.172-.241-.219-.51-.305-.774-.055-.16-.11-.323-.293-.35-.2-.031-.278.136-.356.276-.313.572-.434 1.202-.422 1.84.027 1.436.633 2.58 1.838 3.393.137.093.172.187.129.323-.082.28-.18.552-.266.833-.055.179-.137.217-.329.14a5.526 5.526 0 01-1.736-1.18c-.857-.828-1.631-1.742-2.597-2.458a11.365 11.365 0 00-.689-.471c-.985-.957.13-1.743.388-1.836.27-.098.093-.432-.779-.428-.872.004-1.67.295-2.687.684a3.055 3.055 0 01-.465.137 9.597 9.597 0 00-2.883-.102c-1.885.21-3.39 1.102-4.497 2.623C.082 8.606-.231 10.684.152 12.85c.403 2.284 1.569 4.175 3.36 5.653 1.858 1.533 3.997 2.284 6.438 2.14 1.482-.085 3.133-.284 4.994-1.86.47.234.962.327 1.78.397.63.059 1.236-.03 1.705-.128.735-.156.684-.837.419-.961-2.155-1.004-1.682-.595-2.113-.926 1.096-1.296 2.746-2.642 3.392-7.003.05-.347.007-.565 0-.845-.004-.17.035-.237.23-.256a4.173 4.173 0 001.545-.475c1.396-.763 1.96-2.015 2.093-3.517.02-.23-.004-.467-.247-.588zM11.581 18c-2.089-1.642-3.102-2.183-3.52-2.16-.392.024-.321.471-.235.763.09.288.207.486.371.739.114.167.192.416-.113.603-.673.416-1.842-.14-1.897-.167-1.361-.802-2.5-1.86-3.301-3.307-.774-1.393-1.224-2.887-1.298-4.482-.02-.386.093-.522.477-.592a4.696 4.696 0 011.529-.039c2.132.312 3.946 1.265 5.468 2.774.868.86 1.525 1.887 2.202 2.891.72 1.066 1.494 2.082 2.48 2.914.348.292.625.514.891.677-.802.09-2.14.11-3.054-.614zm1-6.44a.306.306 0 01.415-.287.302.302 0 01.2.288.306.306 0 01-.31.307.303.303 0 01-.304-.308zm3.11 1.596c-.2.081-.399.151-.59.16a1.245 1.245 0 01-.798-.254c-.274-.23-.47-.358-.552-.758a1.73 1.73 0 01.016-.588c.07-.327-.008-.537-.239-.727-.187-.156-.426-.199-.688-.199a.559.559 0 01-.254-.078c-.11-.054-.2-.19-.114-.358.028-.054.16-.186.192-.21.356-.202.767-.136 1.146.016.352.144.618.408 1.001.782.391.451.462.576.685.914.176.265.336.537.445.848.067.195-.019.354-.25.452z" fill="#4D6BFE"/>
    </symbol>
    <symbol id="ico-qwen" viewBox="0 0 24 24">
      <path d="M19.713 2.825h-5.907c0-.251-.066-.503-.196-.728L12.819.728A1.46 1.46 0 0 0 11.559 0h-1.58c-.52 0-1 .275-1.26.728l-2.42 4.19a1.45 1.45 0 0 0 0 1.453l.791 1.37c.26.448.74.727 1.26.727h11.36c.52 0 1-.275 1.26-.728l.79-1.37a1.45 1.45 0 0 0 0-1.451l-.79-1.37a1.46 1.46 0 0 0-1.26-.728zM.195 9.905l2.951 5.115a1.46 1.46 0 0 0-.531.531l-.791 1.37a1.45 1.45 0 0 0 0 1.452l.79 1.37c.26.448.74.727 1.26.727h4.84c.52 0 1-.275 1.26-.728l.79-1.37a1.45 1.45 0 0 0 0-1.451L5.086 7.083a1.46 1.46 0 0 0-1.26-.728h-1.58c-.52 0-1 .276-1.26.728l-.79 1.37a1.45 1.45 0 0 0 0 1.452m15.888 13.367 2.951-5.115c.217.125.469.196.728.196h1.582c.52 0 1-.275 1.26-.728l.79-1.37a1.45 1.45 0 0 0 0-1.451l-2.42-4.191a1.46 1.46 0 0 0-1.259-.728h-1.582c-.52 0-1 .276-1.259.728l-5.678 9.838a1.45 1.45 0 0 0 0 1.452l.79 1.37c.26.448.74.727 1.26.727h1.582c.52 0 1-.275 1.26-.728z" fill="#082dff"/>
    </symbol>
  </defs>
</svg>

<!-- Fixed Opaque Header -->
<header class="nav">
  <div class="wrap nav__bar">
    <a class="nav__logo" href="#top">
      <img src="/logo.svg" width="24" height="24" style="border-radius:6px; flex:none;" alt="">
      <span>声幕 SubCast</span>
      <span class="mono muted" style="font-size:12px; font-weight:400;">2.4</span>
    </a>

    <div class="nav__telemetry">
      <div class="nav__stat">
        <svg class="brand-icon" aria-hidden="true"><use href="#ico-qwen"/></svg>
        <span class="nav__stat-label">语音引擎</span>
        <span id="asrBadge" class="nav__stat-val">Qwen3-ASR-0.6B</span>
      </div>
      <div class="nav__stat">
        <svg id="llmBrandIcon" class="brand-icon" aria-hidden="true"><use id="llmBrandUse" href="#ico-minimax"/></svg>
        <span class="nav__stat-label">翻译模型</span>
        <span id="llmBadge" class="nav__stat-val">...</span>
      </div>
      <div class="nav__stat">
        <span class="nav__stat-label">在线电视</span>
        <span id="tvBadge" class="nav__stat-val">0</span>
      </div>
      <div class="nav__stat">
        <span id="tapDot" class="status-dot"></span>
        <span class="nav__stat-label">监听状态</span>
        <span id="tapBadge" class="nav__stat-val">待机</span>
      </div>
    </div>

    <div class="nav__actions">
      <button type="button" class="btn btn--secondary btn--sm" onclick="toggleTheme()" id="themeBtn">夜间外观</button>
      <a class="btn btn--primary btn--sm" href="/download/lingua-relay-tv.apk">
        下载电视端 APK
        <svg class="m-arrow" viewBox="0 0 12 12" aria-hidden="true"><path d="M1 6h8"/><path d="M4.5 2.5 8 6l-3.5 3.5"/></svg>
      </a>
    </div>
  </div>
</header>

<main id="top" class="wrap main-stack">

  <!-- Section 1: Live Subtitle Monitor -->
  <section class="card">
    <div class="section-head--split">
      <h2>实时字幕监听 <span class="muted">电视端无背景叠加与音频延迟校准</span></h2>
      <div style="display:flex; gap:8px;">
        <button type="button" class="btn btn--secondary btn--sm" onclick="sendTestSub()">发送测试字幕</button>
        <button type="button" class="btn btn--secondary btn--sm" onclick="stopTap()">停止监听</button>
      </div>
    </div>

    <div class="monitor-frame ground-inverse">
      <div class="monitor-bar">
        <div class="monitor-ch">
          <span id="stageDot" class="status-dot"></span>
          <span id="stageChannelTag">待机</span>
          <span id="vuMeter" class="vu-meter" aria-hidden="true" title="音频脉冲状态">
            <i></i><i></i><i></i><i></i><i></i>
          </span>
        </div>
        <div class="monitor-metrics">
          <span id="metricLead">缓冲 --</span>
          <span id="metricPipe">耗时 --</span>
          <span id="metricWait">补偿 --</span>
        </div>
      </div>

      <div class="monitor-stage">
        <div id="subEn" class="sub-en"></div>
        <div id="subZh" class="sub-zh placeholder">等待直播音频流</div>
      </div>
    </div>

    <ul id="historyBox" class="rows" style="display:none; max-height:132px; overflow-y:auto;"></ul>

    <div class="ctrl-bar">
      <div class="ctrl-group">
        <label class="toggle">
          <input type="checkbox" id="autoSyncCheck" checked onchange="saveSettings()">
          <span class="toggle__track"><span class="toggle__thumb"></span></span>
          自动延迟对齐
        </label>
        <div class="ctrl-item">
          <span class="ctrl-label">手动偏移</span>
          <input type="range" id="delaySlider" min="0" max="5000" step="100" value="800" oninput="onDelayInput(this.value)" onchange="saveSettings()">
          <span id="delayVal" class="mono" style="font-size:12.5px; min-width:34px; text-align:right;">0.8s</span>
        </div>
      </div>

      <div class="ctrl-group">
        <div class="ctrl-item">
          <span class="ctrl-label">断句</span>
          <div class="tabs" id="sentenceSeg" role="tablist">
            <button type="button" class="tab" role="tab" aria-selected="true" data-val="full_sentence" onclick="setSentenceMode('full_sentence')">整句</button>
            <button type="button" class="tab" role="tab" aria-selected="false" data-val="paragraph" onclick="setSentenceMode('paragraph')">段落</button>
            <button type="button" class="tab" role="tab" aria-selected="false" data-val="fast_clause" onclick="setSentenceMode('fast_clause')">短句</button>
          </div>
        </div>

        <div class="ctrl-item">
          <span class="ctrl-label">显示</span>
          <div class="tabs" id="modeSeg" role="tablist">
            <button type="button" class="tab" role="tab" aria-selected="true" data-val="bilingual" onclick="setDisplayMode('bilingual')">双语对照</button>
            <button type="button" class="tab" role="tab" aria-selected="false" data-val="zh_only" onclick="setDisplayMode('zh_only')">仅中文</button>
          </div>
        </div>
      </div>
    </div>
  </section>

  <!-- Section 2: Translation Model Config -->
  <section class="card">
    <div class="section-head--split">
      <h2>翻译模型配置 <span class="muted">选择预设或填入接口凭据</span></h2>
      <div class="tabs" id="presetSeg" role="tablist">
        <button type="button" class="tab" role="tab" aria-selected="false" data-model="MiniMax-M3.1-Flash" onclick="applyPreset('https://api.minimaxi.com/v1', 'MiniMax-M3.1-Flash')">
          <svg class="brand-icon" aria-hidden="true"><use href="#ico-minimax"/></svg>
          MiniMax M3.1-Flash 极速
        </button>
        <button type="button" class="tab" role="tab" aria-selected="false" data-model="MiniMax-M3" onclick="applyPreset('https://api.minimaxi.com/v1', 'MiniMax-M3')">
          <svg class="brand-icon" aria-hidden="true"><use href="#ico-minimax"/></svg>
          MiniMax M3
        </button>
        <button type="button" class="tab" role="tab" aria-selected="false" data-model="deepseek-chat" onclick="applyPreset('https://api.deepseek.com/v1', 'deepseek-chat')">
          <svg class="brand-icon" aria-hidden="true"><use href="#ico-deepseek"/></svg>
          DeepSeek
        </button>
      </div>
    </div>

    <div class="api-grid">
      <div class="field">
        <label for="llmBaseUrl">接口地址 (Base URL)</label>
        <input type="text" id="llmBaseUrl" placeholder="https://api.minimaxi.com/v1">
      </div>
      <div class="field">
        <label for="llmApiKey">密钥 (API Key)</label>
        <input type="password" id="llmApiKey" placeholder="sk-...">
      </div>
      <div class="field">
        <label for="llmModel">模型名称 (Model)</label>
        <input type="text" id="llmModel" placeholder="MiniMax-M3.1-Flash" oninput="syncPresetHighlight()">
      </div>
      <button type="button" class="btn btn--primary" onclick="saveAndTestLlm()">保存并测试</button>
    </div>
    <p id="llmTestResult" class="mono" style="font-size:12.5px; display:none;"></p>
  </section>

  <!-- Section 3: Step 01 -> Step 02 & 03 -->
  <div class="grid-12">
    <section class="card span-6">
      <div class="section-head--split">
        <h2><span class="step-num">01</span> 直播源与 EPG 导入 <span class="muted">.txt / .m3u / XMLTV</span></h2>
        <div style="display:flex; gap:8px;">
          <label class="btn btn--secondary btn--sm">
            导入本地文件
            <input type="file" id="fileInput" accept=".txt,.m3u,.m3u8" onchange="loadFile(this)" style="display:none;">
          </label>
          <button type="button" class="btn btn--primary btn--sm" id="btnSavePlaylist" onclick="savePlaylist()">保存并更新</button>
        </div>
      </div>
      <textarea id="playlistEditor" style="height:260px;" placeholder="频道名,http://... 或 #EXTINF..."></textarea>

      <div style="border-top:1px solid var(--line); padding-top:14px; display:grid; gap:10px;">
        <div class="section-head--split">
          <div style="font-size:13px; font-weight:500;">
            在线 EPG 源地址 <span class="muted">点选内置知名源或输入自定义 .xml / .xml.gz</span>
          </div>
          <span id="epgStatusBadge" class="mono muted" style="font-size:12px;">...</span>
        </div>
        <div class="tabs" role="tablist" aria-label="知名 EPG 源预设" id="epgPresetSeg" style="justify-self:start; flex-wrap:wrap;">
          <button type="button" class="tab" role="tab" data-epg="https://epg.112114.xyz/pp.xml.gz,https://live.fanmingming.com/e.xml" aria-selected="true" onclick="applyEpgPreset(this.dataset.epg)">聚合双源 · 推荐</button>
          <button type="button" class="tab" role="tab" data-epg="https://epg.112114.xyz/pp.xml.gz" aria-selected="false" onclick="applyEpgPreset(this.dataset.epg)">112114 极速源</button>
          <button type="button" class="tab" role="tab" data-epg="https://live.fanmingming.com/e.xml" aria-selected="false" onclick="applyEpgPreset(this.dataset.epg)">范明明 Fanmingming</button>
          <button type="button" class="tab" role="tab" data-epg="https://epg.112114.xyz/pp.xml.gz,https://epg.pw/xmltv/epg_HK.xml.gz,https://epg.pw/xmltv/epg_TW.xml.gz,https://epg.pw/xmltv/epg_GB.xml.gz" aria-selected="false" onclick="applyEpgPreset(this.dataset.epg)">epg.pw 港澳台+国际</button>
        </div>
        <div style="display:flex; gap:8px;">
          <input type="text" id="epgSourceInput" class="mono" style="flex:1; height:var(--btn-h-sm); padding:0 10px; border:1px solid var(--line-strong); border-radius:var(--r-sm); background:var(--bg); font-size:12.5px;" placeholder="https://epg.112114.xyz/pp.xml.gz,https://live.fanmingming.com/e.xml" oninput="syncEpgPresetHighlight()">
          <button type="button" class="btn btn--secondary btn--sm" id="btnSyncEpg" onclick="saveAndSyncEpg()">保存并同步 EPG</button>
        </div>
      </div>
    </section>

    <section class="card span-6">
      <div class="section-head--split">
        <h2><span class="step-num">02</span> 代理订阅地址 <span class="muted">填入影视仓或 Jellyfin</span></h2>
      </div>

      <ul class="rows">
        <li>
          <div>
            <div class="rows__name">M3U 订阅 <span class="muted" style="font-size:12px;">推荐 · 影视仓与 Jellyfin 通用，自带台标与 EPG 关联</span></div>
            <div id="m3uUrl" class="rows__sub">...</div>
          </div>
          <button type="button" class="btn btn--secondary btn--sm" onclick="copyText('m3uUrl', this)">复制地址</button>
        </li>
        <li>
          <div>
            <div class="rows__name">EPG 节目单 <span class="muted" style="font-size:12px;">影视仓 DIYP 接口 / Jellyfin XMLTV 指南</span></div>
            <div id="epgUrl" class="rows__sub">...</div>
          </div>
          <button type="button" class="btn btn--secondary btn--sm" onclick="copyText('epgUrl', this)">复制地址</button>
        </li>
        <li>
          <div>
            <div class="rows__name">TXT 纯文本订阅 <span class="muted" style="font-size:12px;">仅含频道与分组</span></div>
            <div id="txtUrl" class="rows__sub">...</div>
          </div>
          <button type="button" class="btn btn--secondary btn--sm" onclick="copyText('txtUrl', this)">复制地址</button>
        </li>
      </ul>

      <div class="section-head--split" style="margin-top:8px;">
        <h2><span class="step-num">03</span> 频道列表 <span id="chCountBadge" class="muted mono">0</span></h2>
        <input type="text" id="chSearchInput" class="search-input" placeholder="筛选频道名称..." oninput="renderChannelList()" aria-label="筛选频道名称">
      </div>

      <ul id="channelList" class="rows ch-scroll"></ul>
    </section>
  </div>

</main>

<script>
let ws = null;
let llmFormInitialized = false;
let allChannels = [];
let activeUpstreamUrl = '';
let isSidecarActive = false;
let subHistory = [];
let currentSentenceMode = 'full_sentence';
let currentDisplayMode = 'bilingual';
let vuPulseTimer = null;

function applyTheme(theme) {
  if (theme === 'dark') {
    document.documentElement.setAttribute('data-tone', 'dark');
    document.getElementById('themeBtn').textContent = '日间外观';
  } else {
    document.documentElement.removeAttribute('data-tone');
    document.getElementById('themeBtn').textContent = '夜间外观';
  }
}

function toggleTheme() {
  const isDark = document.documentElement.getAttribute('data-tone') === 'dark';
  const next = isDark ? 'light' : 'dark';
  localStorage.setItem('lr_theme', next);
  applyTheme(next);
}

applyTheme(localStorage.getItem('lr_theme') || 'light');

function syncPresetHighlight() {
  const m = document.getElementById('llmModel').value.trim();
  document.querySelectorAll('#presetSeg .tab').forEach(btn => {
    btn.setAttribute('aria-selected', btn.dataset.model === m ? 'true' : 'false');
  });
  const useEl = document.getElementById('llmBrandUse');
  if (useEl) {
    useEl.setAttribute('href', m.toLowerCase().includes('deepseek') ? '#ico-deepseek' : '#ico-minimax');
  }
}

function applyPreset(baseUrl, model) {
  document.getElementById('llmBaseUrl').value = baseUrl;
  document.getElementById('llmModel').value = model;
  syncPresetHighlight();
}

function syncEpgPresetHighlight() {
  const inp = document.getElementById('epgSourceInput');
  if (!inp) return;
  const val = inp.value.trim();
  document.querySelectorAll('#epgPresetSeg .tab').forEach(btn => {
    btn.setAttribute('aria-selected', btn.dataset.epg === val ? 'true' : 'false');
  });
}

function applyEpgPreset(url) {
  document.getElementById('epgSourceInput').value = url;
  syncEpgPresetHighlight();
}

function updateSegActive(containerId, val) {
  document.querySelectorAll(`#${containerId} .tab`).forEach(btn => {
    btn.setAttribute('aria-selected', btn.dataset.val === val ? 'true' : 'false');
  });
}

function setSentenceMode(val) {
  currentSentenceMode = val;
  updateSegActive('sentenceSeg', val);
  saveSettings();
}

function setDisplayMode(val) {
  currentDisplayMode = val;
  updateSegActive('modeSeg', val);
  saveSettings();
}

async function saveAndTestLlm() {
  const resEl = document.getElementById('llmTestResult');
  resEl.style.display = 'block';
  resEl.style.color = 'var(--muted)';
  resEl.textContent = '正在验证接口...';
  const payload = {
    enabled: true,
    base_url: document.getElementById('llmBaseUrl').value.trim(),
    api_key: document.getElementById('llmApiKey').value.trim(),
    model: document.getElementById('llmModel').value.trim(),
    source_lang: 'auto'
  };
  const res = await fetch('/api/llm-config', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload)
  });
  const data = await res.json();
  if (data.error) {
    resEl.style.color = 'var(--action)';
    resEl.textContent = `连接失败: ${data.error}`;
  } else {
    resEl.style.color = 'var(--ink)';
    resEl.textContent = `${data.engine_used} · ${data.latency_ms}ms · "${data.sample_zh}"`;
  }
  loadStatus();
}

function renderChannelList() {
  const listEl = document.getElementById('channelList');
  const q = (document.getElementById('chSearchInput').value || '').trim().toLowerCase();
  listEl.innerHTML = '';
  const filtered = allChannels.filter(ch =>
    !q || ch.name.toLowerCase().includes(q) || (ch.group || '').toLowerCase().includes(q)
  );
  document.getElementById('chCountBadge').textContent = `${filtered.length} 个频道`;
  filtered.forEach(ch => {
    const li = document.createElement('li');
    const isCurrent = isSidecarActive && activeUpstreamUrl === ch.url;
    const dot = isCurrent ? '<span class="status-dot is-live"></span>' : '';
    const logo = ch.logo ? `<img class="ch-logo" src="${ch.logo}" alt="" loading="lazy" onerror="this.style.display='none'">` : '';
    const grp = ch.group ? `<span class="muted" style="font-size:12.5px; font-weight:400;">${ch.group}</span>` : '';
    li.innerHTML = `<div class="rows__name">${dot}${logo}<span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${ch.name}</span>${grp}</div>`;
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = isCurrent ? 'btn btn--primary btn--sm' : 'btn btn--secondary btn--sm';
    btn.textContent = isCurrent ? '监听中' : '监听';
    btn.onclick = () => pinChannel(ch.id, ch.name, ch.url);
    li.appendChild(btn);
    listEl.appendChild(li);
  });
}

async function loadStatus() {
  const res = await fetch('/api/status');
  const data = await res.json();
  document.getElementById('txtUrl').textContent = data.txt_url;
  document.getElementById('m3uUrl').textContent = data.m3u_url;
  document.getElementById('epgUrl').textContent = data.epg_xml_url;
  document.getElementById('tvBadge').textContent = data.tv_clients_connected;

  const sc = data.sidecar;
  isSidecarActive = sc.active;
  activeUpstreamUrl = sc.upstream_url || '';
  document.getElementById('asrBadge').textContent = sc.asr_engine;
  document.getElementById('llmBadge').textContent = sc.translation_engine;
  const useEl = document.getElementById('llmBrandUse');
  if (useEl && sc.translation_engine) {
    useEl.setAttribute('href', sc.translation_engine.toLowerCase().includes('deepseek') ? '#ico-deepseek' : '#ico-minimax');
  }

  const tapBadge = document.getElementById('tapBadge');
  const tapDot = document.getElementById('tapDot');
  const stageDot = document.getElementById('stageDot');
  const stageChTag = document.getElementById('stageChannelTag');
  const vuMeter = document.getElementById('vuMeter');
  if (sc.active) {
    tapBadge.textContent = sc.channel_name;
    tapDot.className = 'status-dot is-live';
    stageDot.className = 'status-dot is-live';
    stageChTag.textContent = sc.channel_name;
    vuMeter.classList.add('is-live');
  } else {
    tapBadge.textContent = '待机';
    tapDot.className = 'status-dot';
    stageDot.className = 'status-dot';
    stageChTag.textContent = '待机';
    vuMeter.classList.remove('is-live');
  }

  if (!llmFormInitialized && data.llm_config) {
    document.getElementById('llmBaseUrl').value = data.llm_config.base_url || '';
    document.getElementById('llmModel').value = data.llm_config.model || '';
    if (data.llm_config.api_key_masked) {
      document.getElementById('llmApiKey').value = data.llm_config.api_key_masked;
    }
    if (data.epg_source_url) {
      document.getElementById('epgSourceInput').value = data.epg_source_url;
      syncEpgPresetHighlight();
    }
    syncPresetHighlight();
    llmFormInitialized = true;
  }
  const epgBadge = document.getElementById('epgStatusBadge');
  if (epgBadge) {
    const chCnt = data.epg_channel_count || 0;
    const syncInfo = data.epg_last_sync ? ` · 同步于 ${data.epg_last_sync}` : '';
    epgBadge.textContent = `已缓存 ${chCnt} 频道${syncInfo}`;
  }

  document.getElementById('delaySlider').value = sc.subtitle_delay_ms;
  document.getElementById('delayVal').textContent = (sc.subtitle_delay_ms / 1000).toFixed(1) + 's';
  if (sc.display_mode) {
    currentDisplayMode = sc.display_mode;
    updateSegActive('modeSeg', currentDisplayMode);
  }
  if (sc.sentence_mode) {
    currentSentenceMode = sc.sentence_mode;
    updateSegActive('sentenceSeg', currentSentenceMode);
  }
  if (typeof sc.auto_sync === 'boolean') {
    document.getElementById('autoSyncCheck').checked = sc.auto_sync;
  }

  allChannels = data.channels || [];
  renderChannelList();
}

async function loadRawPlaylist() {
  const res = await fetch('/api/playlist/raw');
  document.getElementById('playlistEditor').value = await res.text();
}

function loadFile(input) {
  const f = input.files[0];
  if (!f) return;
  const reader = new FileReader();
  reader.onload = e => { document.getElementById('playlistEditor').value = e.target.result; };
  reader.readAsText(f, 'utf-8');
}

async function savePlaylist() {
  const btn = document.getElementById('btnSavePlaylist');
  const content = document.getElementById('playlistEditor').value;
  btn.textContent = '更新中...';
  const res = await fetch('/api/playlist', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({content})
  });
  const out = await res.json();
  btn.textContent = `已更新 (${out.channel_count})`;
  setTimeout(() => { btn.textContent = '保存并更新'; }, 1800);
  loadStatus();
}

async function saveAndSyncEpg() {
  const btn = document.getElementById('btnSyncEpg');
  const badge = document.getElementById('epgStatusBadge');
  const url = document.getElementById('epgSourceInput').value.trim();
  if (!url) return;
  btn.textContent = '正在拉取...';
  btn.disabled = true;
  try {
    const res = await fetch('/api/epg-source', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({url})
    });
    const out = await res.json();
    if (out.ok) {
      btn.textContent = '已同步';
      badge.textContent = `已缓存 ${out.channel_count} 频道 (${out.programme_count} 条预告) · ${out.synced_at}`;
    } else {
      btn.textContent = '拉取失败';
      badge.textContent = `失败: ${out.error || '请检查地址'}`;
    }
  } catch (e) {
    btn.textContent = '请求异常';
  } finally {
    btn.disabled = false;
    setTimeout(() => { btn.textContent = '保存并同步 EPG'; }, 2000);
  }
}

function onDelayInput(val) {
  document.getElementById('delayVal').textContent = (val / 1000).toFixed(1) + 's';
}

async function saveSettings() {
  const delay = parseInt(document.getElementById('delaySlider').value, 10);
  const autoSync = document.getElementById('autoSyncCheck').checked;
  await fetch('/api/settings', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      subtitle_delay_ms: delay,
      display_mode: currentDisplayMode,
      sentence_mode: currentSentenceMode,
      auto_sync: autoSync,
      source_lang: 'auto'
    })
  });
  loadStatus();
}

async function pinChannel(id, name, url) {
  await fetch('/api/pin', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({channel_id: id, name, url})
  });
  setTimeout(loadStatus, 350);
}

async function stopTap() {
  await fetch('/api/stop', {method: 'POST'});
  setTimeout(loadStatus, 250);
}

async function sendTestSub() {
  await fetch('/api/test-subtitle', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      en: 'Good evening, breaking news from our live broadcast studio as global markets react to the central bank decision.',
      zh: ''
    })
  });
}

function copyText(id, btn) {
  const t = document.getElementById(id).textContent;
  navigator.clipboard.writeText(t);
  const old = btn.textContent;
  btn.textContent = '已复制';
  setTimeout(() => { btn.textContent = old; }, 1400);
}

function pushHistory(en, zh) {
  if (!zh) return;
  if (subHistory.length > 0 && subHistory[0].zh === zh) return;
  subHistory.unshift({en, zh, time: new Date().toLocaleTimeString()});
  if (subHistory.length > 8) subHistory.pop();
  const box = document.getElementById('historyBox');
  box.style.display = 'block';
  box.innerHTML = subHistory.map(item =>
    `<li>
       <div><span style="font-weight:500; color:var(--ink);">${item.zh}</span> <span class="muted" style="margin-left:8px;">${item.en}</span></div>
       <span class="mono muted" style="font-size:11.5px;">${item.time}</span>
     </li>`
  ).join('');
}

function connectWs() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${proto}//${location.host}/ws/subtitles`);
  ws.onmessage = ev => {
    const msg = JSON.parse(ev.data);
    const zhEl = document.getElementById('subZh');
    if (msg.type === 'subtitle' || msg.type === 'status') {
      if (msg.type === 'subtitle') {
        pushHistory(msg.en || '', msg.zh || '');
        zhEl.className = 'sub-zh';
        const vuEl = document.getElementById('vuMeter');
        if (vuEl && !vuEl.classList.contains('is-live')) {
          vuEl.classList.add('is-live');
          setTimeout(() => {
            const stEl = document.getElementById('stageChannelTag');
            if (stEl && stEl.textContent === '待机') vuEl.classList.remove('is-live');
          }, 2200);
        }
        if (msg.pipeline_elapsed_ms !== undefined) {
          const leadS = ((msg.stream_lead_ms || 3500) / 1000).toFixed(1);
          const pipeS = ((msg.pipeline_elapsed_ms || 0) / 1000).toFixed(2);
          const waitS = ((msg.actual_delay_ms || 0) / 1000).toFixed(2);
          document.getElementById('metricLead').textContent = `缓冲 ${leadS}s`;
          document.getElementById('metricPipe').textContent = `耗时 ${pipeS}s`;
          document.getElementById('metricWait').textContent = `补偿 ${waitS}s`;
        } else if (msg.latency_ms !== undefined) {
          document.getElementById('metricPipe').textContent = `耗时 ${(msg.latency_ms / 1000).toFixed(2)}s`;
        }
      }
      document.getElementById('subEn').textContent = msg.en || '';
      zhEl.textContent = msg.zh || '';
    } else if (msg.type === 'clear') {
      document.getElementById('subEn').textContent = '';
      zhEl.className = 'sub-zh placeholder';
      zhEl.textContent = '已停止监听';
      document.getElementById('metricLead').textContent = '缓冲 --';
      document.getElementById('metricPipe').textContent = '耗时 --';
      document.getElementById('metricWait').textContent = '补偿 --';
    }
  };
  ws.onclose = () => setTimeout(connectWs, 2000);
}

loadStatus();
loadRawPlaylist();
connectWs();
setInterval(loadStatus, 4000);
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def web_console() -> HTMLResponse:
    return HTMLResponse(WEB_CONSOLE_HTML)


@app.get("/logo.svg")
async def get_logo_svg() -> FileResponse:
    logo_svg = ROOT_DIR / "assets" / "logo.svg"
    if not logo_svg.exists():
        raise HTTPException(status_code=404, detail="logo.svg not found")
    return FileResponse(str(logo_svg), media_type="image/svg+xml")


@app.get("/logo.png")
@app.get("/favicon.ico")
async def get_logo_png() -> FileResponse:
    logo_png = ROOT_DIR / "assets" / "logo.png"
    if not logo_png.exists():
        raise HTTPException(status_code=404, detail="logo.png not found")
    return FileResponse(str(logo_png), media_type="image/png")


@app.get("/download/lingua-relay-tv.apk")
async def download_apk() -> FileResponse:
    apk_path = ROOT_DIR / "dist" / "lingua-relay-tv.apk"
    if not apk_path.exists():
        raise HTTPException(status_code=404, detail="APK not built yet")
    return FileResponse(
        path=str(apk_path),
        media_type="application/vnd.android.package-archive",
        filename="lingua-relay-tv.apk",
    )
