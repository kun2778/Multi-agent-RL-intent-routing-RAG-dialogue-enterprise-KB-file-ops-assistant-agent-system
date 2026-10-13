# ============================================================
# AgentRL 桌面版一键打包脚本（PyInstaller onedir）
# 用法：
#   powershell -ExecutionPolicy Bypass -File packaging\build_exe.ps1
#   powershell -File packaging\build_exe.ps1 -SkipVenv      # 用当前环境直接打包
#   powershell -File packaging\build_exe.ps1 -Clean          # 清理旧产物
#   powershell -File packaging\build_exe.ps1 -BundleHF       # 内置 bge 向量模型（离线可用）
# 产物：dist\AgentRL\AgentRL.exe（整个目录即绿色版，可压缩分发）
# ============================================================
param(
    [switch]$SkipVenv,
    [switch]$Clean,
    [switch]$BundleHF,
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot   # packaging/ 的上一级 = 项目根
Set-Location $Root

Write-Host "============================================" -ForegroundColor Cyan
Write-Host " AgentRL Desktop Build" -ForegroundColor Cyan
Write-Host " Project: $Root" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

# ---------- 0. Python 检查 ----------
try {
    $pyVer = & $Python -c "import sys; print('%d.%d' % sys.version_info[:2])"
} catch {
    Write-Host "[ERROR] 找不到 Python: $Python" -ForegroundColor Red
    exit 1
}
Write-Host "[1/7] Python $pyVer ($Python)"

# ---------- 1. 构建专用 venv（推荐：避免污染开发环境，且可装 CPU 版 torch 缩体积） ----------
$venvDir = Join-Path $Root ".build_venv"
$venvPy  = Join-Path $venvDir "Scripts\python.exe"
if (-not $SkipVenv) {
    if (-not (Test-Path $venvPy)) {
        Write-Host "[2/7] 创建构建 venv: $venvDir"
        & $Python -m venv $venvDir
        if ($LASTEXITCODE -ne 0) { exit 1 }
    } else {
        Write-Host "[2/7] 复用已有 venv: $venvDir"
    }
    $Python = $venvPy

    Write-Host "[3/7] 安装依赖（torch 使用 CPU 版以缩小体积，约 200MB vs CUDA 版 2.5GB）..."
    & $Python -m pip install --upgrade pip -q
    # CUDA 版 torch 会让 exe 膨胀到 3GB+；桌面聊天场景 CPU 足够
    & $Python -m pip install torch --index-url https://download.pytorch.org/whl/cpu -q
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[WARN] CPU torch 安装失败，回退 requirements.txt 中的默认 torch" -ForegroundColor Yellow
    }
    & $Python -m pip install -r requirements.txt -q
    if ($LASTEXITCODE -ne 0) { Write-Host "[ERROR] 依赖安装失败" -ForegroundColor Red; exit 1 }
    & $Python -m pip install pyinstaller -q
    if ($LASTEXITCODE -ne 0) { Write-Host "[ERROR] PyInstaller 安装失败" -ForegroundColor Red; exit 1 }
} else {
    Write-Host "[2/7] -SkipVenv：使用当前环境 $Python"
    & $Python -m pip install pyinstaller -q
    Write-Host "[3/7] 跳过依赖安装"
}

# ---------- 2. UPX 检测（可选压缩，最高再省 30%+；找不到就跳过） ----------
$upxDir = $null
$upxCmd = Get-Command upx -ErrorAction SilentlyContinue
if ($upxCmd) {
    $upxDir = Split-Path $upxCmd.Source
} elseif (Test-Path "C:\tools\upx\upx.exe") {
    $upxDir = "C:\tools\upx"
}
if ($upxDir) {
    Write-Host "[4/7] 检测到 UPX: $upxDir（torch 相关 DLL 已在 spec 中排除压缩，避免崩溃）" -ForegroundColor Green
} else {
    Write-Host "[4/7] 未检测到 UPX（可选：从 https://github.com/upx/upx/releases 下载后加入 PATH 可再缩体积）" -ForegroundColor Yellow
}

# ---------- 3. 可选：内置 HuggingFace 向量模型 ----------
if ($BundleHF) {
    Write-Host "[5/7] 下载并内置 BAAI/bge-small-zh-v1.5（~100MB，目标机离线可用）..."
    $env:HF_HOME = Join-Path $Root ".hf_download"
    & $Python -c @"
from sentence_transformers import SentenceTransformer
SentenceTransformer('BAAI/bge-small-zh-v1.5')
print('model cached')
"@
    if ($LASTEXITCODE -ne 0) { Write-Host "[ERROR] 模型下载失败" -ForegroundColor Red; exit 1 }
    $hfCache = Join-Path $env:HF_HOME "hub"
    if (Test-Path $hfCache) {
        if (Test-Path (Join-Path $Root "hf_cache")) { Remove-Item (Join-Path $Root "hf_cache") -Recurse -Force }
        Copy-Item $hfCache (Join-Path $Root "hf_cache") -Recurse
        Write-Host "       已复制到 hf_cache/" -ForegroundColor Green
    }
} else {
    Write-Host "[5/7] 跳过内置向量模型（目标机首次运行会联网下载，或无网时自动回退 TF-IDF）"
}

# ---------- 4. 清理旧产物 ----------
if ($Clean) {
    Write-Host "[6/7] 清理 build/ 与 dist/ ..."
    foreach ($d in @("build", "dist")) {
        $p = Join-Path $Root $d
        if (Test-Path $p) { Remove-Item $p -Recurse -Force }
    }
} else {
    Write-Host "[6/7] 跳过清理（-Clean 可强制清理）"
}

# ---------- 5. 执行打包 ----------
Write-Host "[7/7] PyInstaller 打包中（首次约 5-15 分钟）..." -ForegroundColor Cyan
& $Python -m PyInstaller packaging\AgentRL.spec --noconfirm --clean
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] 打包失败，请检查上方日志" -ForegroundColor Red
    exit 1
}

# ---------- 6. 收尾：复制 .env 模板与说明 ----------
$distDir = Join-Path $Root "dist\AgentRL"
$envSrc  = Join-Path $Root ".env"
$envDst  = Join-Path $distDir ".env"
if ((Test-Path $envSrc) -and (-not (Test-Path $envDst))) {
    Copy-Item $envSrc $envDst
    Write-Host "       已复制 .env（含 API 密钥，请勿外传）" -ForegroundColor Yellow
} elseif (-not (Test-Path $envDst)) {
    Copy-Item (Join-Path $Root ".env.example") $envDst -ErrorAction SilentlyContinue
}

$sizeMB = [math]::Round((Get-ChildItem $distDir -Recurse | Measure-Object Length -Sum).Sum / 1MB, 0)
$readme = @"
AgentRL 桌面版 - 使用说明
================================
1. 双击 AgentRL.exe 启动（自动打开浏览器 http://127.0.0.1:5000/chat.html）
2. API 密钥配置：编辑本目录 .env 文件（DEEPSEEK_API_KEY=... 等）
3. 数据目录：%LOCALAPPDATA%\AgentRL\.runtime（会话/缓存/日志）
4. 模型权重：本目录 models\（可选，缺失时在线学习）
5. 配置文件：本目录 config\*.yaml（可直接修改）

命令行参数：
  AgentRL.exe --no-browser     只启动服务不开浏览器
  AgentRL.exe --port 8080      换端口（默认 5000，被占用自动顺延）

打包体积：约 $sizeMB MB
"@
Set-Content -Path (Join-Path $distDir "使用说明.txt") -Value $readme -Encoding UTF8

Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host " 打包完成！($sizeMB MB)" -ForegroundColor Green
Write-Host " 产物目录: $distDir" -ForegroundColor Green
Write-Host " 分发方式：压缩 dist\AgentRL 整个文件夹为 zip" -ForegroundColor Green
Write-Host " 安装包：  packaging\setup.iss（需安装 Inno Setup 6）" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
