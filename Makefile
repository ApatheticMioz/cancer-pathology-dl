.PHONY: all paper clean clean-paper

all: paper

paper:
	@bash scripts/build_paper.sh

clean-paper:
	@rm -f paper/*.aux paper/*.bbl paper/*.blg paper/*.log paper/*.out paper/*.fls paper/*.fdb_latexmk paper/*.synctex.gz
	@echo "Cleaned LaTeX auxiliary build artifacts."

clean: clean-paper
