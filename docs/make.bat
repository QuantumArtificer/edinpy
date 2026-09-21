@ECHO OFF
set SPHINXBUILD=sphinx-build
if not "%SPHINXBUILD%" == "" goto found
:found
%SPHINXBUILD% -M html . _build
