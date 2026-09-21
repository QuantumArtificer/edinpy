SPHINXBUILD ?= sphinx-build
SPHINXOPTS ?= -W --keep-going
PYTHON ?= python
SOURCEDIR = source
BUILDDIR = _build

.PHONY: html figures clean

html:
	$(SPHINXBUILD) $(SPHINXOPTS) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html"

figures:
	PYTHONPATH=../src $(PYTHON) scripts/generate_figures.py

clean:
	rm -rf "$(BUILDDIR)"
