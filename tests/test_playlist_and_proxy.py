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


def test_foreign_channels_epg_coverage_and_utc8_conversion() -> None:
    from server.main import _to_utc8_xmltv_ts

    # 1. Verify genuine timezone arithmetic to Beijing Time (+0800)
    assert _to_utc8_xmltv_ts("20261112010000 +0000") == "20261112090000 +0800"
    assert _to_utc8_xmltv_ts("20261111200000 -0500") == "20261112090000 +0800"
    assert _to_utc8_xmltv_ts("20261112100000 +0900") == "20261112090000 +0800"
    assert _to_utc8_xmltv_ts("20261112090000 +0800") == "20261112090000 +0800"

    # 2. Verify default M3U header has a single clean x-tvg-url (no comma) and canonical tvg-name
    sample_m3u = """#EXTM3U
#EXTINF:-1 tvg-id="CNN国际新闻" tvg-name="CNN International" group-title="国际/海外",CNN国际新闻 (源1)
http://1.2.3.4/cnn1.m3u8
"""
    parsed = parse_playlist(sample_m3u)
    assert parsed[0].tvg_name == "CNN International"
    exported = export_m3u(parsed, "http://10.10.10.80:8765")
    first_line = exported.splitlines()[0]
    assert 'x-tvg-url="http://10.10.10.80:8765/epg.xml.gz"' in first_line
    assert "," not in first_line
    assert 'tvg-id="CNN国际新闻" tvg-name="CNN国际新闻"' in exported

    # 3. Verify DIYP EPG & XMLTV compatibility routes cover foreign channels (including multi-source & English aliases)
    with TestClient(app) as client:
        r_compat = client.get("/epg.xml.gz,http://10.10.10.80:8765/epg.xml")
        assert r_compat.status_code == 200
        assert len(r_compat.content) > 1000

        for query_ch in [
            "BBC News英国广播公司",
            "CNN国际新闻",
            "CNN International",
            "Al Jazeera English半岛电视台(英语)",
            "LiveNOW from FOX福克斯现场 (源1)",
            "NHK World-Japan日本国际台(英语)",
            "翡翠台",
        ]:
            r_epg = client.get("/epg", params={"ch": query_ch})
            assert r_epg.status_code == 200
            payload = r_epg.json()
            assert len(payload["epg_data"]) > 0, f"Missing EPG for foreign channel: {query_ch}"


def test_multi_client_playlist_and_manifest_tag_propagation() -> None:
    sample = "BBC News,http://example.com/live/bbc.m3u8\nCCTV1,http://example.com/live/cctv1.m3u8\n"
    entries = parse_playlist(sample)
    m3u_tv2 = export_m3u(entries, "http://10.10.10.80:8765", client_id="tv2")
    assert "&client=tv2" in m3u_tv2

    rewritten = rewrite_m3u8_manifest(
        manifest_text="#EXTM3U\n#EXT-X-STREAM-INF:BANDWIDTH=1200000\nchunklist_720p.m3u8\n",
        final_upstream_url="http://example.com/live/bbc.m3u8",
        base_proxy_url="http://10.10.10.80:8765",
        channel_id="bbc01",
        channel_name="BBC News",
        client_id="tv2",
    )
    assert "&client=tv2" in rewritten

    with TestClient(app) as client:
        r_tvbox = client.get("/tv2/tvbox.json")
        assert r_tvbox.status_code == 200
        lives = r_tvbox.json()["lives"]
        assert lives[0]["url"].endswith("/tv2/playlist.m3u")

        r_m3u = client.get("/tv2/playlist.m3u")
        assert r_m3u.status_code == 200
        assert "client=tv2" in r_m3u.text


def test_multi_client_concurrent_isolation_and_same_channel_sharing() -> None:
    import asyncio
    from unittest.mock import patch

    from server.ai_pipeline import SubtitleEvent
    from server.main import get_sidecar
    from server.stream_proxy import AudioSidecarManager

    # Stub out actual FFmpeg subprocess spawning while keeping real pipeline cloning & WebSocket routing
    with (
        patch.object(AudioSidecarManager, "_audio_tap_worker", lambda self, *a, **kw: __import__("time").sleep(0.6)),
        patch.object(AudioSidecarManager, "_translation_worker", lambda self, *a, **kw: None),
        TestClient(app) as client,
    ):
        pool = get_sidecar()
        pool.stop_current()
        pool.auto_sync = False
        pool.subtitle_delay_ms = 0

        with (
            client.websocket_connect("/tv1/ws/subtitles") as ws_tv1,
            client.websocket_connect("/tv2/ws/subtitles") as ws_tv2,
            client.websocket_connect("/ws/subtitles?role=console") as ws_console,
        ):
            conn1 = ws_tv1.receive_json()
            conn2 = ws_tv2.receive_json()
            conn_c = ws_console.receive_json()
            assert conn1["client_key"] == "tag:tv1"
            assert conn2["client_key"] == "tag:tv2"

            # Per-TV display mode isolation: TV 1 switches to zh_only, TV 2 stays bilingual
            ws_tv1.send_json({"action": "update_settings", "display_mode": "zh_only"})
            ack1 = ws_tv1.receive_json()
            assert ack1["type"] == "settings"
            assert ack1["status"]["display_mode"] == "zh_only"

            # TV 1 opens BBC News, TV 2 opens CNN International concurrently
            async def _setup_two_channels():
                sc_bbc = await pool.touch_client_channel(
                    client_key="tag:tv1",
                    client_ip="192.168.31.61",
                    client_tag="tv1",
                    channel_id="bbc_1",
                    channel_name="BBC News (源1)",
                    upstream_url="http://example.com/bbc1.m3u8",
                )
                sc_cnn = await pool.touch_client_channel(
                    client_key="tag:tv2",
                    client_ip="192.168.31.62",
                    client_tag="tv2",
                    channel_id="cnn_1",
                    channel_name="CNN International",
                    upstream_url="http://example.com/cnn1.m3u8",
                )
                return sc_bbc, sc_cnn

            sc_bbc, sc_cnn = pool._loop.run_until_complete(_setup_two_channels()) if not pool._loop.is_running() else asyncio.run_coroutine_threadsafe(_setup_two_channels(), pool._loop).result(timeout=5)

            # Drain initial clear messages sent on new channel start
            assert ws_tv1.receive_json()["type"] == "clear"
            assert ws_tv2.receive_json()["type"] == "clear"

            # Verify isolated LLM context windows & shared ASR lock
            assert sc_bbc is not sc_cnn
            assert sc_bbc.pipeline._shared_asr_lock is sc_cnn.pipeline._shared_asr_lock
            assert sc_bbc.pipeline.llm_translator.context_window is not sc_cnn.pipeline.llm_translator.context_window

            sessions = pool.get_sessions_summary()
            assert len(sessions) == 2

            # Emit a BBC subtitle and a CNN subtitle simultaneously
            sc_bbc._handle_subtitle_event(
                SubtitleEvent(
                    channel="BBC News",
                    en="Good evening from London.",
                    zh="伦敦晚上好。",
                    is_final=True,
                    timestamp=100.0,
                    latency_ms=120.0,
                ),
                session_epoch=sc_bbc._session_epoch,
            )
            sc_cnn._handle_subtitle_event(
                SubtitleEvent(
                    channel="CNN International",
                    en="Live from New York.",
                    zh="纽约现场直播。",
                    is_final=True,
                    timestamp=101.0,
                    latency_ms=130.0,
                ),
                session_epoch=sc_cnn._session_epoch,
            )

            msg_tv1 = ws_tv1.receive_json()
            msg_tv2 = ws_tv2.receive_json()
            assert msg_tv1["zh"] == "伦敦晚上好。"
            assert msg_tv1["display_mode"] == "zh_only"  # TV 1 custom display_mode preserved!
            assert msg_tv2["zh"] == "纽约现场直播。"
            assert msg_tv2["display_mode"] == "bilingual"  # TV 2 unaffected by TV 1!

            # Console receives BOTH subtitles
            console_zhs = {ws_console.receive_json()["zh"], ws_console.receive_json()["zh"]}
            assert console_zhs == {"伦敦晚上好。", "纽约现场直播。"}

            # TV 3 opens BBC News (源2) -> must share existing BBC session (still 2 sessions total!)
            with client.websocket_connect("/tv3/ws/subtitles") as ws_tv3:
                _ = ws_tv3.receive_json()
                sc_bbc_shared = asyncio.run_coroutine_threadsafe(
                    pool.touch_client_channel(
                        client_key="tag:tv3",
                        client_ip="192.168.31.63",
                        client_tag="tv3",
                        channel_id="bbc_2",
                        channel_name="BBC News (源2)",
                        upstream_url="http://example.com/bbc2.m3u8",
                    ),
                    pool._loop,
                ).result(timeout=5)
                assert sc_bbc_shared is sc_bbc
                assert len(pool.get_sessions_summary()) == 2
                # TV 3 receives initial screen clear followed immediately by the replayed latest BBC subtitle!
                assert ws_tv3.receive_json()["type"] == "clear"
                replay_tv3 = ws_tv3.receive_json()
                assert replay_tv3["zh"] == "伦敦晚上好。"

                # TV 2 switches to domestic Chinese channel CCTV-1 -> clears ONLY TV 2 and closes CNN,
                # while BBC session for TV 1 & TV 3 stays 100% active!
                asyncio.run_coroutine_threadsafe(
                    pool.mark_client_chinese_channel(
                        client_key="tag:tv2",
                        client_ip="192.168.31.62",
                        client_tag="tv2",
                        channel_name="CCTV-1综合",
                    ),
                    pool._loop,
                ).result(timeout=5)
                assert ws_tv2.receive_json()["type"] == "clear"
                remaining = pool.get_sessions_summary()
                assert len(remaining) == 1
                assert remaining[0]["channel_key"] == "BBC NEWS"
                assert remaining[0]["viewer_count"] == 2

            pool.stop_current()

        # Also verify zero-config automatic per-IP isolation (/ws/subtitles without tags)
        with (
            client.websocket_connect("/ws/subtitles", headers={"X-Forwarded-For": "192.168.31.71"}) as ws_ip1,
            client.websocket_connect("/ws/subtitles", headers={"X-Forwarded-For": "192.168.31.72"}) as ws_ip2,
        ):
            assert ws_ip1.receive_json()["client_key"] == "ip:192.168.31.71"
            assert ws_ip2.receive_json()["client_key"] == "ip:192.168.31.72"
            sc_sky = asyncio.run_coroutine_threadsafe(
                pool.touch_client_channel(
                    client_key="ip:192.168.31.71",
                    client_ip="192.168.31.71",
                    client_tag="",
                    channel_id="sky_1",
                    channel_name="Sky News",
                    upstream_url="http://example.com/sky.m3u8",
                ),
                pool._loop,
            ).result(timeout=5)
            asyncio.run_coroutine_threadsafe(
                pool.mark_client_chinese_channel(
                    client_key="ip:192.168.31.72",
                    client_ip="192.168.31.72",
                    client_tag="",
                    channel_name="CCTV-13新闻",
                ),
                pool._loop,
            ).result(timeout=5)
            assert ws_ip1.receive_json()["type"] == "clear"
            assert ws_ip2.receive_json()["type"] == "clear"

            sc_sky._handle_subtitle_event(
                SubtitleEvent(
                    channel="Sky News",
                    en="Breaking news from Westminster.",
                    zh="来自威斯敏斯特的突发新闻。",
                    is_final=True,
                    timestamp=200.0,
                    latency_ms=110.0,
                ),
                session_epoch=sc_sky._session_epoch,
            )
            assert ws_ip1.receive_json()["zh"] == "来自威斯敏斯特的突发新闻。"
            pool.stop_current()


def test_precision_pts_closed_loop_sync_and_holdback() -> None:
    import asyncio
    import time
    from unittest.mock import patch

    import httpx

    from server.ai_pipeline import StreamingSpeechPipeline, SubtitleEvent
    from server.main import _seg_cache, get_sidecar
    from server.stream_proxy import (
        AudioSidecarManager,
        ClientPlayheadTracker,
        HlsTimelineClock,
        apply_m3u8_holdback,
        rewrite_m3u8_manifest,
    )

    # 1. Layer 1: Sample-accurate speech-onset timestamping & uncapped long-sentence elapsed calculation
    pipe = StreamingSpeechPipeline()
    pipe.reset_stream(channel_name="Al Jazeera English")
    pipe._stream_samples_cursor = 88000  # 5.5s
    pipe._append_clause_to_pending(
        "In Doha tonight, international mediators announced that.",
        now=1000.0,
        seg_start_sample=32000,  # Exact onset at 2.000s
        seg_end_sample=83200,
    )
    pipe._stream_samples_cursor = 152000  # 9.500s (7.5s of audio elapsed since 2.000s onset!)
    pipe._append_clause_to_pending(
        "Talks will resume tomorrow morning with all delegations present.",
        now=1004.0,
        seg_start_sample=88000,
        seg_end_sample=144000,  # Ends at 9.000s
    )
    ev_long = pipe._flush_pending_to_event()
    assert ev_long is not None
    assert ev_long.audio_onset_sec == 2.0
    assert ev_long.audio_end_sec == 9.0
    assert ev_long.pipeline_elapsed_ms >= 7500.0, (
        f"Long 7.5s utterance must NOT be capped at 1200ms, got {ev_long.pipeline_elapsed_ms}"
    )

    # 2. Layer 2a: Unified HlsTimelineClock across 1080p video & 64k audio variants (PDT + MediaSeq)
    pdt_clock = HlsTimelineClock()
    tv_1080p_m3u8 = """#EXTM3U
#EXT-X-VERSION:3
#EXT-X-TARGETDURATION:6
#EXT-X-MEDIA-SEQUENCE:900
#EXT-X-PROGRAM-DATE-TIME:2026-11-15T12:00:00.000Z
#EXTINF:6.000,
seg_900.ts?sig=tvA
#EXTINF:6.000,
seg_901.ts?sig=tvB
#EXTINF:6.000,
seg_902.ts?sig=tvC
"""
    ff_audio_m3u8 = """#EXTM3U
#EXT-X-VERSION:3
#EXT-X-TARGETDURATION:6
#EXT-X-MEDIA-SEQUENCE:10
#EXT-X-PROGRAM-DATE-TIME:2026-11-15T12:00:06.000Z
#EXTINF:6.000,
aud_10.aac?sig=ffX
#EXTINF:6.000,
aud_11.aac?sig=ffY
"""
    pdt_clock.ingest_manifest("https://cdn.aje.net/1080p/index.m3u8", tv_1080p_m3u8)
    pdt_clock.ingest_manifest("https://cdn.aje.net/audio/index.m3u8", ff_audio_m3u8)
    pts_tv_901 = pdt_clock.lookup_segment_pts("https://cdn.aje.net/1080p/seg_901.ts?sig=fresh_token")
    pts_ff_10 = pdt_clock.lookup_segment_pts("https://cdn.aje.net/audio/aud_10.aac?sig=other_token")
    assert pts_tv_901 is not None and pts_ff_10 is not None
    assert abs(pts_tv_901 - pts_ff_10) < 0.001

    # 3. Layer 3: Adaptive M3U8 Hold-Back Gate (2s short segments vs 10s long segments & >=4 floor)
    short_seg_m3u8 = (
        "#EXTM3U\n#EXT-X-TARGETDURATION:2\n#EXT-X-MEDIA-SEQUENCE:500\n"
        + "".join(f"#EXTINF:2.000,\nchunk_{500 + i}.ts\n" for i in range(9))
    )
    held_m3u8, held_k = apply_m3u8_holdback(short_seg_m3u8, target_lead_sec=9.5)
    assert held_k == 3
    assert "#EXT-X-MEDIA-SEQUENCE:500" in held_m3u8
    assert "chunk_500.ts" in held_m3u8
    assert "chunk_505.ts" in held_m3u8
    assert "chunk_506.ts" not in held_m3u8  # Last 3 tail segments held back cleanly

    aje_10s_m3u8 = (
        "#EXTM3U\n#EXT-X-TARGETDURATION:10\n#EXT-X-MEDIA-SEQUENCE:80\n"
        + "".join(f"#EXTINF:10.000,\naje_{80 + i}.ts\n" for i in range(6))
    )
    _, held_aje = apply_m3u8_holdback(aje_10s_m3u8, target_lead_sec=9.5)
    assert held_aje == 0  # Long 10s segments already have >=10s natural lead -> 0 holdback!

    tiny_4seg_m3u8 = (
        "#EXTM3U\n#EXT-X-TARGETDURATION:2\n#EXT-X-MEDIA-SEQUENCE:1\n"
        + "".join(f"#EXTINF:2.000,\ntiny_{1 + i}.ts\n" for i in range(4))
    )
    _, held_tiny = apply_m3u8_holdback(tiny_4seg_m3u8, target_lead_sec=9.5)
    assert held_tiny == 0  # Never starve playlists with <= 4 segments!

    rewritten_seg_m3u8 = rewrite_m3u8_manifest(
        manifest_text=tv_1080p_m3u8,
        final_upstream_url="https://cdn.aje.net/1080p/index.m3u8",
        base_proxy_url="http://10.10.10.80:8765",
        channel_id="aje01",
        channel_name="Al Jazeera",
        proxy_segments=True,
        client_id="tv1",
        timeline_clock=pdt_clock,
    )
    assert "/live/aje01/seg/seg_900.ts?u=" in rewritten_seg_m3u8
    assert "&seq=900" in rewritten_seg_m3u8
    assert "&dur=6.000" in rewritten_seg_m3u8
    assert "&client=tv1" in rewritten_seg_m3u8

    # 4. Playhead tracker physics: forward pre-fetch stability + causality rebuffer stall shift
    tr = ClientPlayheadTracker(client_key="tag:tv1")
    tr.on_segment_delivered(pts=100.0, dur=4.0, seq=10, done_wall=1000.0)
    w_init = tr.predict_wall_time(110.0)  # Expected: 1000.12 + 10.0 = 1010.12
    assert abs(w_init - 1010.12) < 0.05
    # Immediate burst pre-fetch of seq 11 & 12 at t=1000.25 and t=1000.40 must NOT jump playhead!
    tr.on_segment_delivered(pts=104.0, dur=4.0, seq=11, done_wall=1000.25)
    tr.on_segment_delivered(pts=108.0, dur=4.0, seq=12, done_wall=1000.40)
    assert abs(tr.predict_wall_time(110.0) - w_init) < 0.05
    # Simulated 3.0s network rebuffer stall: seq 13 (pts=112.0) finishes late at t=1015.0 (Playhead 114.88 > 112.0)!
    tr.on_segment_delivered(pts=112.0, dur=4.0, seq=13, done_wall=1015.0)
    assert tr.stall_count == 1
    assert abs(tr.predict_wall_time(114.0) - 1017.08) < 0.05  # Shifted cleanly by the exact stall!

    # 5. End-to-end `/live/{cid}/seg/chunk.ts` pass-through + RAM cache + per-TV closed-loop WS scheduling
    upstream_hits = 0
    fake_ts_bytes = b"\x47" + b"\x00" * 187

    def _mock_handler(req: httpx.Request) -> httpx.Response:
        nonlocal upstream_hits
        upstream_hits += 1
        return httpx.Response(200, content=fake_ts_bytes, headers={"Content-Type": "video/mp2t"})

    mock_async_client = httpx.AsyncClient(transport=httpx.MockTransport(_mock_handler))

    async def _fake_get_http_client(_proxy=None):
        return mock_async_client

    with (
        patch("server.main._get_seg_http_client", _fake_get_http_client),
        patch.object(AudioSidecarManager, "_audio_tap_worker", lambda self, *a, **kw: time.sleep(1.2)),
        patch.object(AudioSidecarManager, "_translation_worker", lambda self, *a, **kw: None),
        TestClient(app) as client,
    ):
        try:
            _seg_cache._store.clear()
            pool = get_sidecar()
            pool.stop_current()
            pool.apply_settings(subtitle_delay_ms=800, auto_sync=True)  # 800ms = neutral 0ms fine-tune

            with (
                client.websocket_connect("/tv1/ws/subtitles") as ws_tv1,
                client.websocket_connect("/tv2/ws/subtitles") as ws_tv2,
            ):
                assert ws_tv1.receive_json()["type"] == "connected"
                assert ws_tv2.receive_json()["type"] == "connected"

                async def _join_both():
                    sc1 = await pool.touch_client_channel(
                        client_key="tag:tv1",
                        client_ip="10.10.10.21",
                        client_tag="tv1",
                        channel_id="aje_live",
                        channel_name="Al Jazeera English",
                        upstream_url="https://cdn.aje.net/live.m3u8",
                    )
                    await pool.touch_client_channel(
                        client_key="tag:tv2",
                        client_ip="10.10.10.22",
                        client_tag="tv2",
                        channel_id="aje_live",
                        channel_name="Al Jazeera English",
                        upstream_url="https://cdn.aje.net/live.m3u8",
                    )
                    return sc1

                sc_aje = asyncio.run_coroutine_threadsafe(_join_both(), pool._loop).result(timeout=5)
                assert ws_tv1.receive_json()["type"] == "clear"
                assert ws_tv2.receive_json()["type"] == "clear"
                # Drain join status message for second viewer
                join_msg = ws_tv2.receive_json()
                assert join_msg["type"] == "status"

                seg_tok = encode_url_token("https://cdn.aje.net/1080p/chunk_700.ts")
                t_ref = time.time()
                # TV1 is at pts=500.0 right now; TV2 is 0.30s behind (pts=499.70 right now)
                r_seg1 = client.get(
                    f"/live/aje_live/seg/chunk_700.ts?u={seg_tok}&pts=500.000&dur=4.000&seq=700&client=tv1"
                )
                r_seg2 = client.get(
                    f"/live/aje_live/seg/chunk_700.ts?u={seg_tok}&pts=499.700&dur=4.000&seq=700&client=tv2"
                )
                assert r_seg1.status_code == 200 and r_seg1.content == fake_ts_bytes
                assert r_seg2.status_code == 200 and r_seg2.content == fake_ts_bytes
                assert upstream_hits == 1, "Second TV fetching same .ts segment should hit SegmentRamCache!"

                # Lock FFmpeg audio origin at pts=500.0; speech onset at +0.25s -> onset_pts = 500.25
                # TV1 reaches 500.25 in ~0.37s (0.12s init + 0.25s); TV2 reaches 500.25 in ~0.67s!
                with sc_aje._lock:
                    sc_aje._audio_origin_pts = 500.0
                    sc_aje._audio_origin_locked = True

                t_emit = time.perf_counter()
                sc_aje._handle_subtitle_event(
                    SubtitleEvent(
                        channel="Al Jazeera English",
                        en="Good evening from Doha, here are tonight's top stories.",
                        zh="多哈晚上好，以下是今晚的头条新闻。",
                        is_final=True,
                        timestamp=t_ref,
                        latency_ms=420.0,
                        pipeline_elapsed_ms=5400.0,
                        audio_onset_sec=0.25,
                        audio_end_sec=4.65,
                    ),
                    session_epoch=sc_aje._session_epoch,
                )

                m1 = ws_tv1.receive_json()
                dt1 = time.perf_counter() - t_emit
                m2 = ws_tv2.receive_json()
                dt2 = time.perf_counter() - t_emit

                assert m1["type"] == "subtitle" and m1["sync_mode"] == "pts_closed_loop"
                assert m2["type"] == "subtitle" and m2["sync_mode"] == "pts_closed_loop"
                assert 0.18 <= dt1 <= 0.60, f"TV1 subtitle arrived at {dt1:.3f}s outside closed-loop window"
                assert (dt2 - dt1) >= 0.18, (
                    f"TV2 (0.30s further buffered) must receive subtitle after TV1: dt1={dt1:.3f}s, dt2={dt2:.3f}s"
                )
        finally:
            pool.stop_current()
            asyncio.run(mock_async_client.aclose())

