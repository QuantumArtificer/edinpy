@ECHO OFF
set SPHINXBUILD=sphinx-build
%SPHINXBUILD% -W --keep-going -b html source _build/html
