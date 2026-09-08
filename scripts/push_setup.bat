@echo off
rem 一键推送 rocky-2k26-cyberface（需先 gh auth login 完成）
chcp 65001 >nul
cd /d "%~dp0.."
set REPO=li281077304-boop/rocky-2k26-cyberface

echo [1/3] 创建私有仓库（若已存在会失败，可忽略）
gh repo create %REPO% --private --source . --push 2>nul
if errorlevel 1 (
  echo [2/3] 仓库可能已存在，尝试补远端
  git remote remove origin 2>nul
  git remote add origin https://github.com/%REPO%.git
)
echo [3/3] 推送 main
git push -u origin main
echo.
echo 完成。验证: gh repo view %REPO%
pause
