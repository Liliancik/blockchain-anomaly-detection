@echo off
REM Run pipeline for each injection fraction

setlocal enabledelayedexpansion

set PYTHON="C:\Users\birsa\OneDrive\Desktop\Kristianstad\Research proj\blockchain-anomaly-detection\.venv\Scripts\python.exe"
set SCRIPT="scripts\run_pipeline.py"

echo Running pipeline for different injection fractions...
echo.

for %%f in (0.01 0.05 0.10 0.20) do (
    echo ================================================================================
    echo Running pipeline with fraction %%f
    echo ================================================================================
    %PYTHON% %SCRIPT% --frac %%f
    echo.
)

echo All runs complete!
