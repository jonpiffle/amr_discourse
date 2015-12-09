LATEX=pdflatex
BIBTEX=bibtex

.PHONY: all paper clean

all: paper

paper: paper.tex paper.bib
	$(LATEX) $@.tex
	$(BIBTEX) $@.aux
	$(LATEX) $@.tex
	$(LATEX) $@.tex

clean:
	find . -name "paper.*" -and \! \( -name "*.tex" -or -name "*.bib" \) -delete
