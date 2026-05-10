# Cross-platform-friendly shortcuts for common project tasks.
# Works on Linux/macOS make and on `make -f Makefile` via Git Bash on Windows.
# PowerShell users can also just run the underlying `python -m ...` commands.

PYTHON ?= python

.PHONY: help install app test lint eval calibrate clean

help:
	@echo "Targets:"
	@echo "  install    - pip install -r requirements.txt"
	@echo "  app        - launch the Streamlit app"
	@echo "  test       - run the pytest suite"
	@echo "  eval       - re-evaluate NLP + CNN on the held-out splits"
	@echo "  calibrate  - search the best NLP decision threshold (no retraining)"
	@echo "  lint       - run pyflakes if available; otherwise byte-compile to surface syntax errors"
	@echo "  clean      - remove __pycache__, .pytest_cache, *.pyc"

install:
	$(PYTHON) -m pip install -r requirements.txt

app:
	$(PYTHON) -m streamlit run app.py

test:
	$(PYTHON) -m pytest

eval:
	$(PYTHON) -m scripts.eval_nlp
	$(PYTHON) -m scripts.eval_cnn --tta

calibrate:
	$(PYTHON) -m scripts.calibrate_nlp_threshold

lint:
	-$(PYTHON) -m pyflakes agents scripts tests system.py app.py streamlit_ui.py streamlit_theme.py config.py text_clean.py training_utils.py
	$(PYTHON) -m compileall -q agents scripts tests system.py app.py streamlit_ui.py streamlit_theme.py config.py text_clean.py training_utils.py

clean:
	$(PYTHON) -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"
	$(PYTHON) -c "import shutil; shutil.rmtree('.pytest_cache', ignore_errors=True)"
