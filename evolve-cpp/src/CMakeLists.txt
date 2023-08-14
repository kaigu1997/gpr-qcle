set(SRC_LIST "evolve.cpp")
set(PY_LIST "main.py" PARENT_SCOPE)
# target python module
pybind11_add_module(evolve SHARED "${SRC_LIST}")
