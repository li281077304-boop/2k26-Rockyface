@echo off
chcp 65001 >nul
title Rocky 8779 Landmark Annotator
cd /d "%~dp0"

set PROFILE=%TEMP%\rkannotator_profile
if exist "%PROFILE%" (
    echo 清理上次遗留的 Chrome profile 锁...
    rd /s /q "%PROFILE%" 2>nul
)

echo Starting HTTP server on http://127.0.0.1:8766/ ...
start "annotator-httpd" cmd /k "%~dp0..\.venv\Scripts\python.exe" -m http.server 8766
timeout /t 2 /nobreak >nul

REM 优先 Chrome (用专用 profile)，找不到则退到默认浏览器
set CHROME=
if exist "C:\Program Files\Google\Chrome\Application\chrome.exe" set CHROME=C:\Program Files\Google\Chrome\Application\chrome.exe
if exist "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" set CHROME=C:\Program Files (x86)\Google\Chrome\Application\chrome.exe

if defined CHROME (
    echo 用 Chrome 打开（专用 profile: %PROFILE%）
    start "" "%CHROME%" --user-data-dir="%PROFILE%" --no-first-run --new-window "http://127.0.0.1:8766/"
) else (
    echo Chrome 没找到，打开默认浏览器
    start "" "http://127.0.0.1:8766/"
)

echo.
echo 浏览器应已启动。如失败，请确认 http://127.0.0.1:8766/ 可手动访问。
echo 关闭前请先关黑色 HTTP 服务窗口。
pause