.PHONY: figures supplement clean all

ROOT := $(abspath ..)
export PYTHONPATH := $(ROOT)

# Only run actual figure/table generators by default (avoid utility CLIs requiring args)
FIG_SCRIPTS := $(wildcard figures/scripts/figure*.py) \
		      $(wildcard figures/scripts/make_table*.py)

figures:
	@echo "Generating figures (PYTHONPATH=$(PYTHONPATH))..."
	@for s in $(FIG_SCRIPTS); do \
		printf "Running %s\n" "$$s"; \
		( cd "$(ROOT)" && python "manuscript/$$s" ) || exit $$?; \
	done

supplement:
	@echo "Generating supplementary figures and tables..."
	@cd "$(ROOT)" && \
		python manuscript/figures/scripts/supplement_confusion_matrices.py && \
		python manuscript/figures/scripts/make_supp_tables.py && \
		python manuscript/figures/scripts/supplement_shap_plots.py && \
		python manuscript/figures/scripts/supplement_seta_perplot_metrics.py && \
		python manuscript/figures/scripts/supplement_threshold_sensitivity.py

all: figures supplement

clean:
	@echo "Cleaning generated outputs..."
	rm -rf figures/output/* experiments/logs/* results/* || true
