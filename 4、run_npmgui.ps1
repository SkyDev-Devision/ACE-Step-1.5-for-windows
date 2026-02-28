# Navigate to UI directory and run setup
# 进入 UI 目录并运行安装脚本
Set-Location ace-step-ui

$VenvPaths = @(
  "./venv/Scripts/activate",
  "./.venv/Scripts/activate",
  "./venv/bin/Activate.ps1",
  "./.venv/bin/activate.ps1"
)

foreach ($Path in $VenvPaths) {
  if (Test-Path $Path) {
    Write-Output "Activating venv: $Path"
    & $Path
    break
  }
}

# Run startup script (OS-aware)
# 运行启动脚本（根据操作系统选择）
if ($IsLinux -or $IsMacOS) {
    if (Test-Path "start.sh") {
        Write-Output "Running start.sh..."
        & bash ./start.sh
    }
    else {
        Write-Warning "start.sh not found"
    }
}
else {
    if (Test-Path "start.bat") {
        Write-Output "Running start.bat..."
        & .\start.bat
    }
    else {
        Write-Warning "start.bat not found"
    }
}

Write-Output "Start finished"
Read-Host | Out-Null ;
