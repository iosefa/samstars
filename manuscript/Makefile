.PHONY: figures clean

FIG_SCRIPTS := $(wildcard figures/scripts/*.py)

figures:
	@echo "Generating figures..."
	@for s in $(FIG_SCRIPTS); do \
		printf "Running %s\n" $$s; \
		python "$$s" || exit $$?; \
	done

clean:
	@echo "Cleaning generated outputs..."
	rm -rf figures/output/* experiments/logs/* results/* || true

