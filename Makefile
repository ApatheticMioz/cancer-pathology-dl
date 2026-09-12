.PHONY: all paper figures clean clean-paper

all: paper

paper:
	@bash scripts/build_paper.sh

# Regenerate the publication figures (fig1 + fig2) via the paper.figures
# package. Uses the project venv if present, otherwise the system python3.
figures:
	@if [ -x venv/bin/python ]; then PY=venv/bin/python; else PY=python3; fi; \
	echo "==> Building figures with $$PY"; \
	$$PY -m paper.figures.build_all

clean-paper:
	@rm -f paper/*.aux paper/*.bbl paper/*.blg paper/*.log paper/*.out paper/*.fls paper/*.fdb_latexmk paper/*.synctex.gz
	@echo "Cleaned LaTeX auxiliary build artifacts."

clean: clean-paper
