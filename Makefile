.PHONY: install install-genai test smoke preview build manipulate run-models evaluate report all clean help

PYTHON ?= python
CONFIG ?= configs/default.yaml

help:
	@echo "Targets:"
	@echo "  install        - install core dependencies"
	@echo "  install-genai  - install optional GenAI dependencies"
	@echo "  test           - run unit tests (excluding slow integration test)"
	@echo "  test-all       - run all tests including slow integration test"
	@echo "  smoke          - run end-to-end smoke test on synthetic data"
	@echo "  preview FACE=path/to/img.jpg - preview manipulations on one image"
	@echo "  build          - build test set manifests (Set A and Set B)"
	@echo "  manipulate     - apply manipulations to Set B"
	@echo "  run-models     - run all enabled models on both sets"
	@echo "  evaluate       - compute metrics from predictions"
	@echo "  report         - generate the HTML report"
	@echo "  all            - run the full pipeline (build->manipulate->run->eval->report)"
	@echo "  clean          - remove generated results / outputs"

install:
	$(PYTHON) -m pip install -r requirements.txt

install-genai:
	$(PYTHON) -m pip install -r requirements-genai.txt

test:
	$(PYTHON) -m pytest tests/ -m "not slow" -q

test-all:
	$(PYTHON) -m pytest tests/ -q

smoke:
	$(PYTHON) -m scripts.smoke_test

preview:
	@if [ -z "$(FACE)" ]; then echo "Usage: make preview FACE=path/to/face.jpg"; exit 1; fi
	$(PYTHON) -m scripts.preview_on_face --image $(FACE)

build:
	$(PYTHON) -m scripts.build_test_sets --config $(CONFIG)

manipulate:
	$(PYTHON) -m scripts.apply_manipulations --config $(CONFIG)

run-models:
	$(PYTHON) -m scripts.run_models --config $(CONFIG)

evaluate:
	$(PYTHON) -m scripts.evaluate --config $(CONFIG)

report:
	$(PYTHON) -m scripts.report --config $(CONFIG)

all:
	$(PYTHON) -m scripts.run_all --config $(CONFIG)

clean:
	rm -rf results/* outputs/* __pycache__ */__pycache__ */*/__pycache__ .pytest_cache
