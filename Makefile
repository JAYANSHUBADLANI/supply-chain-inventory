PYTHON ?= python3

.PHONY: help test phase1 phase2 phase3 phase4 all clean

help:
	@echo "make test    run the test suite"
	@echo "make phase1  clean the raw extract and run the data quality audit"
	@echo "make phase2  demand and lead-time characterisation, ABC/XYZ segmentation"
	@echo "make phase3  safety stock, reorder point, EOQ, service level tradeoff"
	@echo "make phase4  simulation backtest against the naive baseline"
	@echo "make all     run every phase in order"
	@echo "make clean   remove cached interim and processed data"

test:
	$(PYTHON) -m pytest -q

phase1:
	$(PYTHON) run.py 1

phase2:
	$(PYTHON) run.py 2

phase3:
	$(PYTHON) run.py 3

phase4:
	$(PYTHON) run.py 4

all:
	$(PYTHON) run.py all

clean:
	rm -rf data/interim data/processed
	find . -name __pycache__ -type d -exec rm -rf {} +
