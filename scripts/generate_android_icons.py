"""
Generates Android TV & Mobile launcher icons (Adaptive Vector XMLs, Multi-DPI PNGs,
Circular icons, In-App Logo, and 16:9 Android TV Leanback Banner) for 声幕 SubCast.
"""

from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

ROOT_DIR = Path(__file__).resolve().parent.parent
RES_DIR = ROOT_DIR / "android-tv-client" / "app" / "src" / "main" / "res"
ASSETS_DIR = ROOT_DIR / "assets"


def render_master_icon(size: int = 1024, circular: bool = False) -> Image.Image:
    """Renders the high-resolution SubCast hardware icon variant with 4x supersampling."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    s = size / 512.0

    # 1. Outer Chassis Base (#18191B) & Tactile Bevel Border (#2E3035)
    pad = int(round(16 * s))
    inner_pad = int(round(20 * s))
    radius = (size // 2 - pad) if circular else int(round(108 * s))
    inner_radius = (size // 2 - inner_pad) if circular else int(round(104 * s))

    draw.rounded_rectangle(
        [pad, pad, size - pad, size - pad],
        radius=radius,
        fill=(24, 25, 27, 255),
    )

    # Subtle ambient glow behind the center peak bar & primary subtitle track
    glow_layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    gdraw = ImageDraw.Draw(glow_layer)
    gdraw.ellipse(
        [int(164 * s), int(64 * s), int(348 * s), int(288 * s)],
        fill=(255, 74, 31, 44),
    )
    gdraw.rounded_rectangle(
        [int(96 * s), int(338 * s), int(416 * s), int(398 * s)],
        radius=int(24 * s),
        fill=(255, 74, 31, 36),
    )
    glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(radius=int(26 * s)))

    # Clip glow alpha to the chassis mask and alpha-composite onto dark chassis
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        [pad, pad, size - pad, size - pad],
        radius=radius,
        fill=255,
    )
    r_ch, g_ch, b_ch, a_ch = glow_layer.split()
    glow_layer.putalpha(ImageChops.multiply(a_ch, mask))
    img = Image.alpha_composite(img, glow_layer)
    draw = ImageDraw.Draw(img)

    # Anodized hardware rim stroke
    draw.rounded_rectangle(
        [inner_pad, inner_pad, size - inner_pad, size - inner_pad],
        radius=inner_radius,
        outline=(48, 51, 58, 255),
        width=max(2, int(round(4 * s))),
    )

    # 2. Broadcast Horizon Grid Hairline + Micro Calibration Ticks
    y_grid = int(round(316 * s))
    draw.line(
        [(int(88 * s), y_grid), (int(424 * s), y_grid)],
        fill=(44, 47, 54, 255),
        width=max(2, int(round(3.5 * s))),
    )
    for tx in (130, 193, 256, 319, 382):
        x_tick = int(round(tx * s))
        draw.line(
            [(x_tick, int(309 * s)), (x_tick, int(323 * s))],
            fill=(64, 68, 76, 255),
            width=max(1, int(round(2.5 * s))),
        )

    # 3. Upper Zone: 5-Bar Live Broadcast Audio Waveform
    bars = [
        # (x, y, w, h, rgba)
        (112, 172, 36, 96, (238, 238, 176, 196)),
        (175, 116, 36, 152, (238, 238, 236, 255)),
        (238, 84, 36, 184, (255, 74, 31, 255)),   # Signal Orange Center Peak
        (301, 132, 36, 136, (238, 238, 236, 255)),
        (364, 184, 36, 84, (238, 238, 176, 196)),
    ]
    for bx, by, bw, bh, color in bars:
        x0 = int(round(bx * s))
        y0 = int(round(by * s))
        x1 = int(round((bx + bw) * s))
        y1 = int(round((by + bh) * s))
        r = int(round((bw / 2.0) * s))
        draw.rounded_rectangle([x0, y0, x1, y1], radius=r, fill=color)

    # 4. Lower Zone: Synchronized Bilingual Subtitle Tracks
    # Primary Translated Caption Track (Signal Orange #FF4A1F)
    draw.rounded_rectangle(
        [int(104 * s), int(352 * s), int(408 * s), int(386 * s)],
        radius=int(17 * s),
        fill=(255, 74, 31, 255),
    )
    # Secondary Original Caption Track (Titanium #94979C)
    draw.rounded_rectangle(
        [int(148 * s), int(404 * s), int(312 * s), int(426 * s)],
        radius=int(11 * s),
        fill=(148, 151, 156, 255),
    )
    # Live Sync Indicator Dot (Signal Orange #FF4A1F)
    cx, cy, cr = int(348 * s), int(415 * s), int(11 * s)
    draw.ellipse([cx - cr, cy - cr, cx + cr, cy + cr], fill=(255, 74, 31, 255))

    return img


def render_tv_banner(width: int = 320, height: int = 180) -> Image.Image:
    """Renders a 16:9 Android TV Leanback Launcher Banner (rendered at 1280x720, downscaled to 320x180)."""
    W, H = 1280, 720
    banner = Image.new("RGBA", (W, H), (20, 21, 24, 255))

    # Subtle ambient orange glow on left-center
    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    gdraw = ImageDraw.Draw(glow)
    gdraw.ellipse([int(W * 0.03), int(H * 0.08), int(W * 0.42), int(H * 0.92)], fill=(255, 74, 31, 38))
    glow = glow.filter(ImageFilter.GaussianBlur(radius=64))
    banner = Image.alpha_composite(banner, glow)
    draw = ImageDraw.Draw(banner)

    # Subtle outer border frame
    draw.rounded_rectangle(
        [10, 10, W - 10, H - 10],
        radius=44,
        outline=(48, 51, 58, 255),
        width=4,
    )

    # Paste high-res SubCast icon on the left side
    icon_size = 470
    icon_img = render_master_icon(size=icon_size, circular=False)
    icon_x = 76
    icon_y = (H - icon_size) // 2
    banner.alpha_composite(icon_img, (icon_x, icon_y))

    # Typography on the right side
    text_x = icon_x + icon_size + 52

    def _load_font(candidates: list[str], pt: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        for c in candidates:
            p = Path(c)
            if p.exists():
                try:
                    return ImageFont.truetype(str(p), pt)
                except Exception:
                    pass
        return ImageFont.load_default()

    cjk_bold = [
        "C:/Windows/Fonts/msyhbd.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    ]
    cjk_reg = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    mono_bold = [
        "C:/Windows/Fonts/consolab.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
    ]

    font_title = _load_font(cjk_bold, 88)
    font_sub = _load_font(cjk_reg, 36)
    font_badge = _load_font(mono_bold, 26)

    # "AI LIVE SUBTITLES" top pill badge
    pill_y = 168
    draw.rounded_rectangle(
        [text_x, pill_y, text_x + 336, pill_y + 52],
        radius=26,
        fill=(44, 24, 20, 255),
        outline=(255, 74, 31, 210),
        width=3,
    )
    draw.ellipse([text_x + 22, pill_y + 18, text_x + 38, pill_y + 34], fill=(255, 74, 31, 255))
    draw.text((text_x + 54, pill_y + 11), "AI LIVE SUBTITLES", fill=(255, 110, 74, 255), font=font_badge)

    # Main Title: 声幕 SubCast
    draw.text((text_x, 265), "声幕 SubCast", fill=(244, 244, 246, 255), font=font_title)

    # Subtitle description (using CJK font)
    draw.text(
        (text_x, 405),
        "电视直播多语种实时同传字幕",
        fill=(162, 166, 174, 255),
        font=font_sub,
    )

    # Bottom accent bars matching the logo
    bar_y = 508
    draw.rounded_rectangle([text_x, bar_y, text_x + 310, bar_y + 18], radius=9, fill=(255, 74, 31, 255))
    draw.rounded_rectangle([text_x + 330, bar_y, text_x + 500, bar_y + 18], radius=9, fill=(116, 120, 128, 255))
    draw.ellipse([text_x + 524, bar_y, text_x + 542, bar_y + 18], fill=(255, 74, 31, 255))

    return banner.resize((width, height), Image.Resampling.LANCZOS)


def write_adaptive_icon_xmls() -> None:
    """Writes Android 8.0+ (API 26) Adaptive Icon vector drawables and mipmap XMLs."""
    drawable_dir = RES_DIR / "drawable"
    anydpi_dir = RES_DIR / "mipmap-anydpi-v26"
    drawable_dir.mkdir(parents=True, exist_ok=True)
    anydpi_dir.mkdir(parents=True, exist_ok=True)

    bg_xml = """<?xml version="1.0" encoding="utf-8"?>
<vector xmlns:android="http://schemas.android.com/apk/res/android"
    android:width="108dp"
    android:height="108dp"
    android:viewportWidth="512"
    android:viewportHeight="512">
    <path
        android:fillColor="#18191B"
        android:pathData="M0,0h512v512h-512z" />
    <path
        android:strokeColor="#2A2C30"
        android:strokeWidth="3"
        android:pathData="M96,316L416,316" />
</vector>
"""

    fg_xml = """<?xml version="1.0" encoding="utf-8"?>
<vector xmlns:android="http://schemas.android.com/apk/res/android"
    android:width="108dp"
    android:height="108dp"
    android:viewportWidth="512"
    android:viewportHeight="512">
    <group
        android:pivotX="256"
        android:pivotY="256"
        android:scaleX="0.68"
        android:scaleY="0.68">
        <!-- Broadcast Horizon Line -->
        <path
            android:strokeColor="#2E3138"
            android:strokeWidth="4"
            android:strokeLineCap="round"
            android:pathData="M88,316L424,316" />
        <!-- Bar 1 -->
        <path
            android:fillColor="#EEEEB0"
            android:fillAlpha="0.78"
            android:pathData="M130,172L130,172A18,18 0,0 1,148 190L148,250A18,18 0,0 1,130 268L130,268A18,18 0,0 1,112 250L112,190A18,18 0,0 1,130 172z" />
        <!-- Bar 2 -->
        <path
            android:fillColor="#EEEEEC"
            android:pathData="M193,116L193,116A18,18 0,0 1,211 134L211,250A18,18 0,0 1,193 268L193,268A18,18 0,0 1,175 250L175,134A18,18 0,0 1,193 116z" />
        <!-- Bar 3 (Signal Orange Peak) -->
        <path
            android:fillColor="#FF4A1F"
            android:pathData="M256,84L256,84A18,18 0,0 1,274 102L274,250A18,18 0,0 1,256 268L256,268A18,18 0,0 1,238 250L238,102A18,18 0,0 1,256 84z" />
        <!-- Bar 4 -->
        <path
            android:fillColor="#EEEEEC"
            android:pathData="M319,132L319,132A18,18 0,0 1,337 150L337,250A18,18 0,0 1,319 268L319,268A18,18 0,0 1,301 250L301,150A18,18 0,0 1,319 132z" />
        <!-- Bar 5 -->
        <path
            android:fillColor="#EEEEB0"
            android:fillAlpha="0.78"
            android:pathData="M382,184L382,184A18,18 0,0 1,400 202L400,250A18,18 0,0 1,382 268L382,268A18,18 0,0 1,364 250L364,202A18,18 0,0 1,382 184z" />
        <!-- Primary Translated Subtitle Track -->
        <path
            android:fillColor="#FF4A1F"
            android:pathData="M121,352L391,352A17,17 0,0 1,408 369L408,369A17,17 0,0 1,391 386L121,386A17,17 0,0 1,104 369L104,369A17,17 0,0 1,121 352z" />
        <!-- Secondary Original Subtitle Track -->
        <path
            android:fillColor="#94979C"
            android:pathData="M159,404L301,404A11,11 0,0 1,312 415L312,415A11,11 0,0 1,301 426L159,426A11,11 0,0 1,148 415L148,415A11,11 0,0 1,159 404z" />
        <!-- Live Sync Dot -->
        <path
            android:fillColor="#FF4A1F"
            android:pathData="M348,415m-11,0a11,11 0,1 1,22 0a11,11 0,1 1,-22 0" />
    </group>
</vector>
"""

    adaptive_xml = """<?xml version="1.0" encoding="utf-8"?>
<adaptive-icon xmlns:android="http://schemas.android.com/apk/res/android">
    <background android:drawable="@drawable/ic_launcher_background" />
    <foreground android:drawable="@drawable/ic_launcher_foreground" />
</adaptive-icon>
"""

    (drawable_dir / "ic_launcher_background.xml").write_text(bg_xml, encoding="utf-8")
    (drawable_dir / "ic_launcher_foreground.xml").write_text(fg_xml, encoding="utf-8")
    (anydpi_dir / "ic_launcher.xml").write_text(adaptive_xml, encoding="utf-8")
    (anydpi_dir / "ic_launcher_round.xml").write_text(adaptive_xml, encoding="utf-8")


def generate_all() -> None:
    master_sq = render_master_icon(size=1024, circular=False)
    master_rd = render_master_icon(size=1024, circular=True)

    # Also update assets/logo.png with the enhanced 512x512 master icon
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    master_sq.resize((512, 512), Image.Resampling.LANCZOS).save(ASSETS_DIR / "logo.png", "PNG")

    densities = {
        "mipmap-mdpi": 48,
        "mipmap-hdpi": 72,
        "mipmap-xhdpi": 96,
        "mipmap-xxhdpi": 144,
        "mipmap-xxxhdpi": 192,
    }
    for folder, px in densities.items():
        out_dir = RES_DIR / folder
        out_dir.mkdir(parents=True, exist_ok=True)
        master_sq.resize((px, px), Image.Resampling.LANCZOS).save(out_dir / "ic_launcher.png", "PNG")
        master_rd.resize((px, px), Image.Resampling.LANCZOS).save(out_dir / "ic_launcher_round.png", "PNG")
        print(f"[Icon] Generated {folder}/ic_launcher.png & ic_launcher_round.png ({px}x{px})")

    # In-app header logo (drawable-nodpi/ic_logo.png)
    nodpi_dir = RES_DIR / "drawable-nodpi"
    nodpi_dir.mkdir(parents=True, exist_ok=True)
    master_sq.resize((192, 192), Image.Resampling.LANCZOS).save(nodpi_dir / "ic_logo.png", "PNG")
    print("[Icon] Generated drawable-nodpi/ic_logo.png (192x192)")

    # Android TV Leanback 16:9 Banner (drawable-xhdpi/tv_banner.png)
    xhdpi_draw_dir = RES_DIR / "drawable-xhdpi"
    xhdpi_draw_dir.mkdir(parents=True, exist_ok=True)
    banner_img = render_tv_banner(width=320, height=180)
    banner_img.save(xhdpi_draw_dir / "tv_banner.png", "PNG")
    print("[Icon] Generated drawable-xhdpi/tv_banner.png (320x180 Leanback TV Banner)")

    # Android 8.0+ Adaptive Icon XMLs
    write_adaptive_icon_xmls()
    print("[Icon] Generated Adaptive Icon XMLs (mipmap-anydpi-v26 & drawable)")


if __name__ == "__main__":
    generate_all()
