.PHONY: figures clean

ROOT := $(abspath ..)
export PYTHONPATH := $(ROOT)

FIG_SCRIPTS := $(wildcard figures/scripts/*.py)

figures:
	@echo "Generating figures (PYTHONPATH=$(PYTHONPATH))..."
	@for s in $(FIG_SCRIPTS); do \
		dir=$$(dirname "$$s"); base=$$(basename "$$s"); \
		printf "Running %s\n" "$$s"; \
		( cd "$$dir" && python "$$base" ) || exit $$?; \
	done

clean:
	@echo "Cleaning generated outputs..."
	rm -rf figures/output/* experiments/logs/* results/* || true
