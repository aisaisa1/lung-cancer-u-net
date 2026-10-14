@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo IQ-OTH/NCCD Lung Q1 - OpenCode Research Assistant
echo ============================================================
echo.
echo Project directory:
echo %CD%
echo.

where opencode >nul 2>nul
if errorlevel 1 (
    echo [ERROR] opencode was not found in PATH.
    echo Install/configure OpenCode first, then run this BAT again.
    pause
    exit /b 1
)

if not exist "OPENCODE_MASTER_PROMPT.md" (
    echo [ERROR] OPENCODE_MASTER_PROMPT.md is missing from this directory.
    pause
    exit /b 1
)

if not exist "ROADMAP_Q1_LUNG_SEGMENTATION.md" (
    echo [ERROR] ROADMAP_Q1_LUNG_SEGMENTATION.md is missing from this directory.
    pause
    exit /b 1
)

echo Starting OpenCode in the current project.
echo The prompt tells OpenCode to STOP after the teacher feasibility gate.
echo.
echo NOTE: This launcher does NOT use --auto.
echo You can review permission requests before edits/commands execute.
echo.

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$prompt = Get-Content -Raw -LiteralPath '.\OPENCODE_MASTER_PROMPT.md'; opencode run --agent build $prompt"

echo.
echo ============================================================
echo OpenCode session finished.
echo Review its report and OUTPUT_HASIL before continuing.
echo ============================================================
pause
endlocal
