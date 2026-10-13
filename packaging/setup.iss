; ============================================================
; AgentRL 桌面版 Inno Setup 安装脚本
; 前置：先运行 packaging\build_exe.ps1 生成 dist\AgentRL\
; 然后用 Inno Setup 6 编译本文件生成 setup.exe
;   iscc packaging\setup.iss
; ============================================================

#define MyAppName "AgentRL 智能助手"
#define MyAppVersion "2.0.0"
#define MyAppPublisher "AgentRL Project"
#define MyAppExeName "AgentRL.exe"
#define DistDir "..\dist\AgentRL"

[Setup]
AppId={{D8F3C2A1-6E4B-4E8A-9C7D-1B2F3A4C5D6E}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\AgentRL
DefaultGroupName={#MyAppName}
; 用户级安装：无需管理员权限，且 %LOCALAPPDATA% 数据目录始终可写
PrivilegesRequired=lowest
OutputDir=.
OutputBaseFilename=AgentRL-Setup-{#MyAppVersion}
; 如需图标：放置 packaging\app.ico 后取消下一行注释
; SetupIconFile=app.ico
; 未签名 exe 时降低 SmartScreen 干扰（治本仍是代码签名）
SignedUninstaller=no
Compression=lzma2/max
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64
WizardStyle=modern
DisableProgramGroupPage=yes
; 安装后提示重启浏览器（服务端口被占用时）
CloseApplications=yes

[Languages]
Name: "chinesesimplified"; MessagesFile: "compiler:Default.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; 整个 onedir 产物目录（含 exe、DLL、数据文件、config、models）
Source: "{#DistDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
; VC++ 运行库合并安装（x64），彻底避免目标机缺 DLL
; 下载：https://aka.ms/vs/17/release/vc_redist.x64.exe 放到 packaging\ 目录
Source: "vc_redist.x64.exe"; DestDir: "{tmp}"; Flags: skipifsourcedoesntexist

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\卸载 {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
; 先装 VC++ 运行库（静默），再启动应用
Filename: "{tmp}\vc_redist.x64.exe"; Parameters: "/install /quiet /norestart"; StatusMsg: "正在安装 Visual C++ 运行库..."; Flags: skipifdoesntexist
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#MyAppName}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; 卸载时清理运行时数据（保守：只删缓存日志，保留会话）
Type: filesandordirs; Name: "{localappdata}\AgentRL\.runtime\cache"
Type: filesandordirs; Name: "{localappdata}\AgentRL\.runtime\temp"
Type: filesandordirs; Name: "{localappdata}\AgentRL\.runtime\logs"
