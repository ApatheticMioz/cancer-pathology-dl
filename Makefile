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

# CBM graphical abstract: a wide ~2.5:1 banner (Claimed-vs-Measured / Why /
# Audit protocol) built by the paper.figures package. It overwrites
# paper/graphical_abstract.{pdf,png}; the paper itself embeds nothing of it.
graphical-abstract:
	@if [ -x venv/bin/python ]; then PY=venv/bin/python; else PY=python3; fi; \
	echo "==> Building graphical abstract (wide CBM banner) with $$PY"; \
	$$PY -m paper.figures.build_all ga

clean-paper:
	@rm -f paper/*.aux paper/*.bbl paper/*.blg paper/*.log paper/*.out paper/*.fls paper/*.fdb_latexmk paper/*.synctex.gz
	@echo "Cleaned LaTeX auxiliary build artifacts."

clean: clean-paper
