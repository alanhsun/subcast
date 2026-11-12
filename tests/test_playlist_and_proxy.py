from __future__ import annotations

from fastapi.testclient import TestClient

from server.main import app
from server.playlist_parser import (
    decode_url_token,
    encode_url_token,
    export_m3u,
    export_txt,
    parse_playlist,
)
from server.stream_proxy import rewrite_m3u8_manifest


def test_parse_tvbox_txt_and_export() -> None:
    sample_txt = """
新闻频道,#genre#
BBC World,https://example.com/live/bbc.m3u8#https://backup.example.com/bbc.ts
CNN Live,http://1.2.3.4:8080/cnn/mpegts$HD
"""
    channels = parse_playlist(sample_txt)
    assert len(channels) == 3
    assert channels[0].name == "BBC World (源1)"
    assert channels[0].is_hls is True
    assert channels[1].name == "BBC World (源2)"
    assert channels[1].is_hls is False
    assert channels[2].name == "CNN Live"
    assert channels[2].url == "http://1.2.3.4:8080/cnn/mpegts"

    m3u_out = export_m3u(channels, "http://10.10.10.80:8765")
    assert "#EXTM3U" in m3u_out
    assert "http://10.10.10.80:8765/live/" in m3u_out
    assert "/index.m3u8?u=" in m3u_out
    assert "/stream.ts?u=" in m3u_out

    txt_out = export_txt(channels, "http://10.10.10.80:8765")
    assert "新闻频道,#genre#" in txt_out
    assert "CNN Live,http://10.10.10.80:8765/live/" in txt_out


def test_parse_m3u_format() -> None:
    sample_m3u = """#EXTM3U
#EXTINF:-1 tvg-id="BBC1" tvg-name="BBC One" group-title="UK TV",BBC One HD
https://cdn.example.org/hls/bbcone/master.m3u8
"""
    channels = parse_playlist(sample_m3u)
    assert len(channels) == 1
    assert channels[0].name == "BBC One HD"
    assert channels[0].group == "UK TV"
    assert channels[0].is_hls is True


def test_rewrite_hls_manifest() -> None:
    master_m3u8 = """#EXTM3U
#EXT-X-STREAM-INF:BANDWIDTH=2500000,RESOLUTION=1280x720
720p/prog_index.m3u8
#EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID="aac",NAME="English",URI="audio/en.m3u8"
"""
    rewritten_master = rewrite_m3u8_manifest(
        manifest_text=master_m3u8,
        final_upstream_url="https://stream.example.com/live/master.m3u8",
        base_proxy_url="http://10.10.10.80:8765",
        channel_id="ch01",
        channel_name="BBC",
    )
    assert "http://10.10.10.80:8765/live/ch01/index.m3u8?u=" in rewritten_master

    media_m3u8 = """#EXTM3U
#EXT-X-TARGETDURATION:4
#EXTINF:4.0,
seg_001.ts
#EXTINF:4.0,
/abs/seg_002.ts
"""
    rewritten_media = rewrite_m3u8_manifest(
        manifest_text=media_m3u8,
        final_upstream_url="https://stream.example.com/live/720p/prog_index.m3u8",
        base_proxy_url="http://10.10.10.80:8765",
        channel_id="ch01",
        channel_name="BBC",
    )
    assert "https://stream.example.com/live/720p/seg_001.ts" in rewritten_media
    assert "https://stream.example.com/abs/seg_002.ts" in rewritten_media


def test_fastapi_endpoints_and_websocket() -> None:
    with TestClient(app) as client:
        r_status = client.get("/api/status")
        assert r_status.status_code == 200
        data = r_status.json()
        assert "m3u_url" in data
        assert "txt_url" in data

        r_m3u = client.get("/playlist.m3u")
        assert r_m3u.status_code == 200
        assert "#EXTM3U" in r_m3u.text

        r_txt = client.get("/playlist.txt")
        assert r_txt.status_code == 200
        assert ",#genre#" in r_txt.text

        with client.websocket_connect("/ws/subtitles") as ws:
            hello = ws.receive_json()
            assert hello["type"] == "connected"

            # Trigger a test subtitle and verify WebSocket client receives it with Chinese translation
            r_test = client.post(
                "/api/test-subtitle",
                json={"en": "breaking news from london tonight", "zh": ""},
            )
            assert r_test.status_code == 200
            sub_msg = ws.receive_json()
            assert sub_msg["type"] == "subtitle"
            assert "London" in sub_msg["en"] or "london" in sub_msg["en"].lower()
            assert len(sub_msg["zh"]) > 0
            print("Received WS test subtitle:", sub_msg)


def test_foreign_vs_chinese_channel_classification_and_epg() -> None:
    sample_m3u = """#EXTM3U
#EXTINF:-1 tvg-id="CCTV1" tvg-name="CCTV1" group-title="央视频道",CCTV1
http://1.2.3.4/cctv1.m3u8
#EXTINF:-1 tvg-id="CGTN" tvg-name="CGTN" group-title="央视频道",CGTN
http://1.2.3.4/cgtn.m3u8
#EXTINF:-1 tvg-id="凤凰中文" tvg-name="凤凰中文" group-title="香港/澳门",凤凰中文
http://1.2.3.4/fhzw.m3u8
#EXTINF:-1 tvg-id="翡翠台" tvg-name="翡翠台" group-title="香港/澳门",翡翠台
http://1.2.3.4/jade.m3u8
#EXTINF:-1 tvg-id="Sky News天空新闻" tvg-name="Sky News" group-title="国际/海外",Sky News天空新闻
http://1.2.3.4/sky.m3u8
#EXTINF:-1 tvg-id="Love Nature 4K" tvg-name="Love Nature 4K" group-title="4K超高清",Love Nature 4K
http://1.2.3.4/lovenature.m3u8
"""
    channels = parse_playlist(sample_m3u)
    by_name = {c.name: c for c in channels}
    assert by_name["CCTV1"].need_translate is False
    assert by_name["凤凰中文"].need_translate is False
    assert by_name["翡翠台"].need_translate is True
    assert by_name["CGTN"].need_translate is True
    assert by_name["Sky News天空新闻"].need_translate is True
    assert by_name["Love Nature 4K"].need_translate is True

    m3u_all = export_m3u(
        channels,
        "http://10.10.10.80:8765",
        extra_epg_url="https://epg.112114.xyz/pp.xml.gz",
    )
    assert 'x-tvg-url="http://10.10.10.80:8765/epg.xml.gz,http://10.10.10.80:8765/epg.xml,https://epg.112114.xyz/pp.xml.gz"' in m3u_all

    m3u_foreign = export_m3u(channels, "http://10.10.10.80:8765", only_foreign=True)
    assert "CCTV1" not in m3u_foreign
    assert "凤凰中文" not in m3u_foreign
    assert "翡翠台" in m3u_foreign
    assert "CGTN" in m3u_foreign
    assert "Sky News天空新闻" in m3u_foreign


def test_epg_and_foreign_playlist_routes() -> None:
    with TestClient(app) as client:
        r_epg = client.get("/epg.xml")
        assert r_epg.status_code == 200
        assert "<tv" in r_epg.text
        assert "<channel" in r_epg.text

        r_epg_gz = client.get("/epg.xml.gz")
        assert r_epg_gz.status_code == 200
        assert len(r_epg_gz.content) > 1000

        r_diyp = client.get("/epg?ch=CCTV1")
        assert r_diyp.status_code == 200
        diyp_json = r_diyp.json()
        assert diyp_json["channel_name"] == "CCTV1"
        assert isinstance(diyp_json["epg_data"], list)
        assert len(diyp_json["epg_data"]) > 0

        r_logo_svg = client.get("/logo.svg")
        assert r_logo_svg.status_code == 200
        assert "<svg" in r_logo_svg.text

        r_logo_png = client.get("/logo.png")
        assert r_logo_png.status_code == 200
        assert len(r_logo_png.content) > 1000

        r_apk = client.get("/download/subcast-tv.apk")
        assert r_apk.status_code == 200
        assert len(r_apk.content) > 10000

        r_foreign = client.get("/playlist_foreign.m3u")
        assert r_foreign.status_code == 200
        assert "CGTN" in r_foreign.text
        assert "CCTV1," not in r_foreign.text


def test_channel_switch_cancels_delayed_subtitles_and_switches_sources() -> None:
    import asyncio
    from unittest.mock import MagicMock

    from server.ai_pipeline import SubtitleEvent
    from server.stream_proxy import (
        AudioSidecarManager,
        _discover_sibling_master_urls,
        _rank_audio_candidates,
    )

    # 1. Verify sibling Master M3U8 discovery from high-bitrate single-variant URLs
    assert "https://cdn.example.com/live/master.m3u8" in _discover_sibling_master_urls(
        "https://cdn.example.com/live/master_6500000.m3u8"
    )
    assert "https://cdn.example.com/dw/master.m3u8" in _discover_sibling_master_urls(
        "https://cdn.example.com/dw/01.m3u8"
    )
    assert "https://cdn.example.com/fox/master.m3u8" in _discover_sibling_master_urls(
        "https://cdn.example.com/fox/highw/index.m3u8"
    )

    # 2. Verify slow URL demotion and Master M3U8 preference
    ranked = _rank_audio_candidates(
        "https://slow.ro/liveibu5_highw/index.m3u8",
        [
            "https://slow.ro/liveibu5_highw/index.m3u8",
            "https://fast.amagi.tv/playlist.m3u8",
        ],
        slow_urls={"https://slow.ro/liveibu5_highw/index.m3u8"},
    )
    assert ranked[0] == "https://fast.amagi.tv/playlist.m3u8"

    # 3. Verify delayed subtitle from previous channel is 100% cancelled after channel switch or stop
    async def _run_epoch_test() -> None:
        received: list[dict] = []

        async def _cb(msg: dict) -> None:
            received.append(msg)

        fake_pipeline = MagicMock()
        fake_pipeline.sentence_mode = "full_sentence"
        sidecar = AudioSidecarManager(pipeline=fake_pipeline, broadcast_cb=_cb)
        sidecar.attach_loop(asyncio.get_running_loop())

        try:
            old_epoch = sidecar._session_epoch
            ev_old = SubtitleEvent(
                channel="Al Jazeera",
                en="Previous channel sentence that was delayed by auto_sync.",
                zh="上一频道因延迟补偿尚在排队等待发送的旧字幕。",
                is_final=True,
                timestamp=1000.0,
                latency_ms=400.0,
                pipeline_elapsed_ms=900.0,
            )
            sidecar.auto_sync = False
            sidecar.subtitle_delay_ms = 220
            sidecar._handle_subtitle_event(ev_old, session_epoch=old_epoch)

            # Switch channel / stop after 40ms (while ev_old is still sleeping for 220ms)
            await asyncio.sleep(0.04)
            sidecar.stop_current()
            await asyncio.sleep(0.28)

            # Old channel's subtitle must NEVER arrive or overwrite _last_subtitle_payload!
            leaked = [m for m in received if m.get("type") == "subtitle"]
            assert leaked == [], f"Stale delayed subtitle leaked after channel switch: {leaked}"
            assert sidecar._last_subtitle_payload is None

            # 4. Verify switching between Source 1 and Source 2 of same channel name updates active_channel_id
            sidecar._audio_tap_worker = lambda *a, **kw: __import__("time").sleep(0.4)  # type: ignore[method-assign]
            sidecar._translation_worker = lambda *a, **kw: None  # type: ignore[method-assign]

            sidecar.touch_channel(
                channel_id="src1_id",
                channel_name="LiveNOW from FOX福克斯现场 (源1)",
                upstream_url="https://slow.ro/index.m3u8",
            )
            assert sidecar.active_channel_id == "src1_id"

            sidecar.touch_channel(
                channel_id="src2_id",
                channel_name="LiveNOW from FOX福克斯现场 (源2)",
                upstream_url="https://fast.amagi.tv/playlist.m3u8",
            )
            assert sidecar.active_channel_id == "src2_id"
            assert sidecar.active_upstream_url == "https://fast.amagi.tv/playlist.m3u8"
        finally:
            sidecar.stop_current()
            if sidecar._watchdog_task:
                sidecar._watchdog_task.cancel()

    asyncio.run(_run_epoch_test())
