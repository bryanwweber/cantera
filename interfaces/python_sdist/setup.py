import sys
import os
from setuptools import setup, Extension
from setuptools.command.install import install
from setuptools.command.develop import develop
from pathlib import Path
import numpy
import shutil

HERE = Path(__file__).parent
CT_SRC = HERE / "src"
EXT_SRC = HERE / "ext"
CT_INCLUDE = HERE / "include"
BOOST_INCLUDE = None
FORCE_CYTHON_COMPILE = False

CYTHON_BUILT_FILES = [HERE / "cantera" / f"_cantera.{ext}" for ext in ("cpp", "h")]


class CanteraOptionsMixin:
    """Custom options for the install and develop commands.

    Modeled after https://stackoverflow.com/a/53833930
    """
    user_options = [
        ("force-cython-compile", None, "Force compilation of .pyx files via Cython"),
        ("boost-include", None, "Location of the Boost header files."),
    ]

    def initialize_options(self):
        super().initialize_options()
        self.force_cython_compile = False
        self.boost_include = None

    def finalize_options(self):
        if self.boost_include is not None:
            assert Path(self.boost_include).is_dir(), f"The path {self.boost_include!r} is not a directory."
        super().finalize_options()

    def run(self):
        global BOOST_INCLUDE, FORCE_CYTHON_COMPILE
        BOOST_INCLUDE = self.boost_include
        FORCE_CYTHON_COMPILE = self.force_cython_compile
        super().run()


class InstallCommand(CanteraOptionsMixin, install):
    user_options = getattr(install, "user_options", []) + CanteraOptionsMixin.user_options


class DevelopCommand(CanteraOptionsMixin, develop):
    user_options = getattr(develop, "user_options", []) + CanteraOptionsMixin.user_options


if not all(p.exists() for p in CYTHON_BUILT_FILES) or "sdist" in sys.argv or FORCE_CYTHON_COMPILE or os.environ.get("FORCE_CYTHON_COMPILE", False):
    from Cython.Build import cythonize
    CYTHON_EXT = ".pyx"
    for p in CYTHON_BUILT_FILES:
        if p.exists():
            p.unlink()
else:
    CYTHON_EXT = ".cpp"
    def cythonize(extensions):
        """Define a no-op for when we"re not using Cython."""
        return extensions

source_files = ["cantera/_cantera" + CYTHON_EXT]
source_files += list(map(str, CT_SRC.glob("**/*.cpp")))
sundials_sources = list(map(str, EXT_SRC.glob("sundials/**/*.c")))
yaml_cpp_sources = list(map(str, EXT_SRC.glob("yaml-cpp/**/*.cpp")))
fmt_sources = list(map(str, EXT_SRC.glob("fmt/*.cc")))
libexecstream_sources = [str(EXT_SRC / "libexecstream" / "exec-stream.cpp")]

include_dirs = [
    str(CT_INCLUDE),
    str(CT_INCLUDE / "cantera" / "ext"),
    str(CT_SRC),
    numpy.get_include()
]

if "BOOST_INCLUDE" in os.environ:
    include_dirs.append(os.environ["BOOST_INCLUDE"])
elif BOOST_INCLUDE is not None:
    include_dirs.append(BOOST_INCLUDE)

if sys.platform != "win32":
    extra_compile_args = ["-std=c++11"]
    sundials_configh = {
        "SUNDIALS_USE_GENERIC_MATH": "#define SUNDIALS_USE_GENERIC_MATH 1",
        "SUNDIALS_BLAS_LAPACK": "/* #undef SUNDIALS_BLAS_LAPACK */"
    }
else:
    extra_compile_args = []
    sundials_configh = {
        "SUNDIALS_USE_GENERIC_MATH": "/* #undef SUNDIALS_USE_GENERIC_MATH */",
        "SUNDIALS_BLAS_LAPACK": "/* #undef SUNDIALS_BLAS_LAPACK */"
    }

config_h_in = (HERE / "sundials_config.h.in").read_text()
config_h = HERE / "sundials_config.h"
config_h.write_text(config_h_in.format_map(sundials_configh))
shutil.copy2(config_h, EXT_SRC / "sundials" / "sundials")
shutil.copy2(config_h, CT_INCLUDE / "cantera" / "ext" / "sundials")

extensions = cythonize([
    Extension(
        "cantera._cantera",
        source_files,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        language="c++",
    ),
])

libraries = [
    ("sundials", {"sources": sundials_sources, "cflags": ["-w"], "include_dirs": include_dirs + [str(EXT_SRC / "sundials" / "sundials")]}),
    ("yaml-cpp", {"sources": yaml_cpp_sources, "cflags": ["-std=c++11"], "include_dirs": include_dirs}),
    ("fmtlib", {"sources": fmt_sources, "cflags": ["-std=c++11"], "include_dirs": include_dirs}),
    ("libexecstream", {"sources": libexecstream_sources, "include_dirs": include_dirs})
]

try:
    setup(
        ext_modules=extensions,
        libraries=libraries,
        cmdclass={"install": InstallCommand, "develop": DevelopCommand},
    )
except Exception:
    from textwrap import dedent
    message = """\
        Building failed, you moron.
    """
