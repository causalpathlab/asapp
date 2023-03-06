from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

__version__ = "0.0.1"

ext_modules = [
    Pybind11Extension("asapc",
        ["src/main.cpp","src/cpp_asap.cpp"],
        include_dirs=[
            "include/",
            "${EIGEN3_INCLUDE_DIR}"
        ],
        # Example: passing in the version to the compiled code
        define_macros = [('VERSION_INFO', __version__)],
        language='c++'
        ),
]

setup(
    name="asapc",
    version=__version__,
    ext_modules=ext_modules,
    extras_require={"test": "pytest"},
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
)