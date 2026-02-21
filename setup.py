# setup.py
from Cython.Build import cythonize
from setuptools import setup
from setuptools.extension import Extension

# Create the 'compiled' directory if it doesn't exist
# os.makedirs("compiled", exist_ok=True)

# Define the extension module
extensions = [
    Extension(
        "potions.hydro",  # the module name in python
        ["src/potions/hydro.py"],  # the cython source
        include_dirs=["."],
        define_macros=[("CYTHON_PROFILE", "1")],
    ),
    Extension(
        "potions.math",  # the module name in python
        ["src/potions/math.py"],  # the cython source
        include_dirs=["."],
        define_macros=[("CYTHON_PROFILE", "1")],
    ),
]

for ext in extensions:
    ext.py_limited_api = False
    # ext.define_macros += [("Py_LIMITED_API", None)]

setup(
    name="potions-model",
    ext_modules=cythonize(extensions, build_dir="compiled", annotate=True),
    # script_args=["build_ext", "--inplace"], # This may not be neccesary depending on system
    zip_safe=False,
)
