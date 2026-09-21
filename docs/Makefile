SPHINXBUILD ?= sphinx-build
PYTHON ?= python
SOURCEDIR = source
BUILDDIR = _build

.PHONY: html figures clean

html:
	$(SPHINXBUILD) -W -b html "$(SOURCEDIR)" "$(BUILDDIR)/html"

figures:
	PYTHONPATH=../src $(PYTHON) scripts/generate_figures.py

clean:
	rm -rf "$(BUILDDIR)"
