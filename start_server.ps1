$ErrorActionPreference = "Stop"
$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RootDir

$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = $RootDir
$env:NO_PROXY = "*"
$env:no_proxy = "*"

Write-Host "=== 声桥 TV 版 (LinguaRelay-TV) 本地服务端启动 ===" -ForegroundColor Cyan
python scripts/download_models.py
Write-Host "正在启动网关服务: http://0.0.0.0:8765 ..." -ForegroundColor Green
python -m uvicorn server.main:app --host 0.0.0.0 --port 8765
