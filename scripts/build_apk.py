"""
Self-contained Android TV APK compiler for LinguaRelay-TV (声桥 TV 版).
Automatically provisions portable OpenJDK 17, Android build-tools (aapt2, d8, zipalign, apksigner),
Android platform android.jar, and platform-tools (adb.exe) into `.toolchain/`, then builds and signs
`dist/lingua-relay-tv.apk`.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
TOOLCHAIN_DIR = ROOT_DIR / ".toolchain"
ANDROID_PROJECT_DIR = ROOT_DIR / "android-tv-client" / "app" / "src" / "main"
BUILD_TMP = ROOT_DIR / "android-tv-client" / "build"
DIST_DIR = ROOT_DIR / "dist"

JDK_URL = "https://aka.ms/download-jdk/microsoft-jdk-17-windows-x64.zip"
PLATFORM_TOOLS_URL = "https://dl.google.com/android/repository/platform-tools-latest-windows.zip"
BUILD_TOOLS_URL = "https://dl.google.com/android/repository/build-tools_r34-windows.zip"
PLATFORM_JAR_URL = "https://dl.google.com/android/repository/platform-34-ext8_r01.zip"


def download_and_extract(url: str, extract_to: Path, marker_glob: str, label: str) -> Path:
    matches = list(extract_to.rglob(marker_glob))
    if matches:
        print(f"[OK] {label} found: {matches[0]}")
        return matches[0]

    extract_to.mkdir(parents=True, exist_ok=True)
    zip_path = TOOLCHAIN_DIR / f"{label}.zip"
    if not zip_path.exists():
        print(f"[Download] {label} <- {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "LinguaRelay-TV-Builder/1.0"})
        with urllib.request.urlopen(req, timeout=180) as resp, open(zip_path, "wb") as out:
            total = int(resp.headers.get("Content-Length", 0))
            done = 0
            last_pct = -1
            while True:
                chunk = resp.read(512 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                if total > 0:
                    pct = int(done * 100 / total)
                    if pct >= last_pct + 20 or done == total:
                        last_pct = pct
                        print(f"  -> {label}: {pct}% ({done / 1048576:.1f} / {total / 1048576:.1f} MB)")

    print(f"[Extract] Extracting {label}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    zip_path.unlink(missing_ok=True)

    matches = list(extract_to.rglob(marker_glob))
    if not matches:
        raise RuntimeError(f"Could not find {marker_glob} inside {extract_to}")
    return matches[0]


def run_cmd(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print("[Run]", " ".join(str(c) for c in cmd))
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if res.returncode != 0:
        print("STDOUT:", res.stdout)
        print("STDERR:", res.stderr)
        raise RuntimeError(f"Command failed with exit code {res.returncode}: {cmd[0]}")


def build_apk() -> Path:
    TOOLCHAIN_DIR.mkdir(parents=True, exist_ok=True)
    DIST_DIR.mkdir(parents=True, exist_ok=True)

    javac_exe = download_and_extract(JDK_URL, TOOLCHAIN_DIR / "jdk", "javac.exe", "OpenJDK-17")
    jdk_bin = javac_exe.parent
    jdk_home = jdk_bin.parent
    java_exe = jdk_bin / "java.exe"
    keytool_exe = jdk_bin / "keytool.exe"

    adb_exe = download_and_extract(
        PLATFORM_TOOLS_URL, TOOLCHAIN_DIR / "platform-tools", "adb.exe", "Android-Platform-Tools"
    )
    aapt2_exe = download_and_extract(
        BUILD_TOOLS_URL, TOOLCHAIN_DIR / "build-tools", "aapt2.exe", "Android-Build-Tools-34"
    )
    build_tools_dir = aapt2_exe.parent
    zipalign_exe = build_tools_dir / "zipalign.exe"
    d8_jar = build_tools_dir / "lib" / "d8.jar"
    apksigner_jar = build_tools_dir / "lib" / "apksigner.jar"

    android_jar = download_and_extract(
        PLATFORM_JAR_URL, TOOLCHAIN_DIR / "platform-34", "android.jar", "Android-Platform-34"
    )

    if BUILD_TMP.exists():
        shutil.rmtree(BUILD_TMP)
    classes_dir = BUILD_TMP / "classes"
    dex_dir = BUILD_TMP / "dex"
    classes_dir.mkdir(parents=True, exist_ok=True)
    dex_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["JAVA_HOME"] = str(jdk_home)
    env["PATH"] = f"{jdk_bin};{build_tools_dir};{adb_exe.parent};{env.get('PATH', '')}"

    manifest_path = ANDROID_PROJECT_DIR / "AndroidManifest.xml"
    unaligned_apk = BUILD_TMP / "unaligned.apk"
    aligned_apk = BUILD_TMP / "aligned.apk"
    final_apk = DIST_DIR / "lingua-relay-tv.apk"

    # 1. Link manifest with aapt2 to create base APK container and R.java (if any)
    r_java_dir = BUILD_TMP / "gen"
    r_java_dir.mkdir(parents=True, exist_ok=True)
    run_cmd(
        [
            str(aapt2_exe),
            "link",
            "-o",
            str(unaligned_apk),
            "-I",
            str(android_jar),
            "--manifest",
            str(manifest_path),
            "--java",
            str(r_java_dir),
        ],
        env=env,
    )

    # 2. Compile Java sources (targeting Java 8 bytecode for universal Android compatibility)
    java_sources = list((ANDROID_PROJECT_DIR / "java").rglob("*.java")) + list(
        r_java_dir.rglob("*.java")
    )
    run_cmd(
        [
            str(javac_exe),
            "-encoding",
            "UTF-8",
            "--release",
            "8",
            "-classpath",
            str(android_jar),
            "-d",
            str(classes_dir),
            *[str(s) for s in java_sources],
        ],
        env=env,
    )

    # 3. Convert .class files to classes.dex using d8
    class_files = list(classes_dir.rglob("*.class"))
    run_cmd(
        [
            str(java_exe),
            "-cp",
            str(d8_jar),
            "com.android.tools.r8.D8",
            "--lib",
            str(android_jar),
            "--min-api",
            "21",
            "--output",
            str(dex_dir),
            *[str(c) for c in class_files],
        ],
        env=env,
    )

    # 4. Add classes.dex into unaligned.apk
    classes_dex = dex_dir / "classes.dex"
    with zipfile.ZipFile(unaligned_apk, "a", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(classes_dex, "classes.dex")

    # 5. Zipalign
    run_cmd([str(zipalign_exe), "-f", "-p", "4", str(unaligned_apk), str(aligned_apk)], env=env)

    # 6. Generate debug/release signing keystore if not exists and sign with apksigner (v1 + v2 + v3)
    keystore_path = TOOLCHAIN_DIR / "shengqiao_tv.keystore"
    if not keystore_path.exists():
        run_cmd(
            [
                str(keytool_exe),
                "-genkeypair",
                "-v",
                "-keystore",
                str(keystore_path),
                "-alias",
                "shengqiao",
                "-keyalg",
                "RSA",
                "-keysize",
                "2048",
                "-validity",
                "10000",
                "-storepass",
                "shengqiao123",
                "-keypass",
                "shengqiao123",
                "-dname",
                "CN=LinguaRelayTV, OU=AI, O=ShengQiao, L=CN, S=CN, C=CN",
            ],
            env=env,
        )

    run_cmd(
        [
            str(java_exe),
            "-jar",
            str(apksigner_jar),
            "sign",
            "--ks",
            str(keystore_path),
            "--ks-key-alias",
            "shengqiao",
            "--ks-pass",
            "pass:shengqiao123",
            "--key-pass",
            "pass:shengqiao123",
            "--out",
            str(final_apk),
            str(aligned_apk),
        ],
        env=env,
    )

    # 7. Verify APK signature
    run_cmd([str(java_exe), "-jar", str(apksigner_jar), "verify", "-v", str(final_apk)], env=env)

    size_kb = final_apk.stat().st_size / 1024.0
    print(f"=== Successfully built & signed APK: {final_apk} ({size_kb:.1f} KB) ===")
    print(f"=== ADB binary ready at: {adb_exe} ===")
    return final_apk


if __name__ == "__main__":
    build_apk()
