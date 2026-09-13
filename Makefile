.PHONY: all paper figures graphical-abstract clean clean-paper

all: paper

paper:
	@bash scripts/build_paper.sh

# Regenerate the publication figures (fig1 + fig2) via the paper.figures
# package. Uses the project venv if present, otherwise the system python3.
figures:
	@if [ -x venv/bin/python ]; then PY=venv/bin/python; else PY=python3; fi; \
	echo "==> Building figures with $$PY"; \
	$$PY -m paper.figures.build_all

# CBM graphical abstract: standalone copy of the results dot plot so the
# provenance (which figure is submitted as the graphical abstract) is explicit.
graphical-abstract:
	@cp paper/fig3_results_dotplot.pdf paper/graphical_abstract.pdf
	@echo "==> paper/graphical_abstract.pdf (copy of paper/fig3_results_dotplot.pdf)"

clean-paper:
	@rm -f paper/*.aux paper/*.bbl paper/*.blg paper/*.log paper/*.out paper/*.fls paper/*.fdb_latexmk paper/*.synctex.gz
	@echo "Cleaned LaTeX auxiliary build artifacts."

clean: clean-paper
