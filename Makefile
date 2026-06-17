.PHONY: install browser test smoke baselines

install:
	pip install -r requirements.txt

browser:            ## one-time: download headless Chromium
	playwright install chromium

test:               ## browserless unit tests (no Chromium)
	pytest -q

smoke:              ## run the example agent against the mock
	python -m harness.evaluate --agent example --mock

baselines:          ## run floor + heuristic against headless Chromium
	python -m harness.evaluate --agent random
	python -m harness.evaluate --agent probe --budget 400
