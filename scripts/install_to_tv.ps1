param(
    [Parameter(Mandatory=$false)]
    [string]$TvIp = "192.168.31.63"
)

$ErrorActionPreference = "Stop"
$RootDir = Split-Path -Parent $PSScriptRoot
$ApkPath = Join-Path $RootDir "dist\lingua-relay-tv.apk"
$Adb = Join-Path $RootDir ".toolchain\platform-tools\platform-tools\adb.exe"

if (-not (Test-Path $Adb)) {
    $FallbackAdb = "$env:LOCALAPPDATA\Packages\OpenAI.Codex_2p2nqsd0c76g0\LocalCache\Local\mobile-mcp\android\platform-tools\adb.exe"
    if (Test-Path $FallbackAdb) {
        $Adb = $FallbackAdb
    } else {
        $Adb = "adb"
    }
}

if (-not (Test-Path $ApkPath)) {
    Write-Host "[1/4] Building Android TV APK..." -ForegroundColor Cyan
    python (Join-Path $RootDir "scripts\build_apk.py")
}

Write-Host "[2/4] Connecting to TV ($TvIp:5555)..." -ForegroundColor Cyan
& $Adb connect "$TvIp`:5555"

Write-Host "[3/4] Installing lingua-relay-tv.apk to $TvIp..." -ForegroundColor Cyan
& $Adb -s "$TvIp`:5555" push "$ApkPath" /sdcard/Download/lingua-relay-tv.apk
& $Adb -s "$TvIp`:5555" install -r "$ApkPath"

Write-Host "[4/4] Granting SYSTEM_ALERT_WINDOW & APP_AUTO_START permissions..." -ForegroundColor Cyan
& $Adb -s "$TvIp`:5555" shell appops set com.shengqiao.tv SYSTEM_ALERT_WINDOW allow
& $Adb -s "$TvIp`:5555" shell cmd appops set com.shengqiao.tv APP_AUTO_START allow
& $Adb -s "$TvIp`:5555" shell am start -n com.shengqiao.tv/.MainActivity

Write-Host "=== Done! 声幕 SubCast is running on $TvIp ===" -ForegroundColor Green
