# PowerShell equivalent of the Makefile.
#
# Usage (from project root, with venv activated):
#     .\tasks.ps1 app          # launch Streamlit
#     .\tasks.ps1 test         # run pytest
#     .\tasks.ps1 eval         # eval NLP + CNN (with TTA)
#     .\tasks.ps1 calibrate    # calibrate NLP threshold
#     .\tasks.ps1 lint         # byte-compile (catches syntax errors)
#     .\tasks.ps1 install      # pip install -r requirements.txt
#     .\tasks.ps1 clean        # remove __pycache__ + .pytest_cache
#     .\tasks.ps1 help         # this help
#
# If PowerShell blocks the script with an execution-policy error, run once:
#     Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

param(
    [Parameter(Position = 0)]
    [string]$task = "help"
)

$python = "python"

switch ($task.ToLower()) {
    "help" {
        Write-Host "Available tasks:"
        Write-Host "  app          - launch Streamlit"
        Write-Host "  test         - run pytest"
        Write-Host "  eval         - re-evaluate NLP + CNN on held-out splits"
        Write-Host "  calibrate    - search best NLP decision threshold"
        Write-Host "  lint         - byte-compile to surface syntax errors"
        Write-Host "  install      - pip install -r requirements.txt"
        Write-Host "  clean        - remove __pycache__ and .pytest_cache"
    }
    "install" {
        & $python -m pip install -r requirements.txt
    }
    "app" {
        & $python -m streamlit run app.py
    }
    "test" {
        & $python -m pytest
    }
    "eval" {
        & $python -m scripts.eval_nlp
        & $python -m scripts.eval_cnn --tta
    }
    "calibrate" {
        & $python -m scripts.calibrate_nlp_threshold
    }
    "lint" {
        & $python -m compileall -q agents scripts tests system.py app.py streamlit_ui.py streamlit_theme.py config.py text_clean.py training_utils.py
    }
    "clean" {
        Get-ChildItem -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue |
            ForEach-Object { Remove-Item -Recurse -Force $_.FullName }
        if (Test-Path ".pytest_cache") { Remove-Item -Recurse -Force ".pytest_cache" }
        Write-Host "cleaned __pycache__ and .pytest_cache"
    }
    default {
        Write-Host "Unknown task: $task"
        Write-Host "Run '.\tasks.ps1 help' for the list."
        exit 1
    }
}
