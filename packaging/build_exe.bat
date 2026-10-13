@echo off
REM 双击即可打包（自动用管理员权限无关的 PowerShell 执行）
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0build_exe.ps1" %*
pause
