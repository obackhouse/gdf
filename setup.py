#!/usr/bin/env python3

import os
import sys
import glob
import shlex
import shutil
from setuptools import setup, find_packages, Extension, Command
from setuptools.command.test import test
from setuptools.command.build_ext import build_ext

setup_src = os.path.dirname(os.path.realpath(__file__))


class CMakeExtension(Extension):
    """Initialise the name of a CMake extension."""

    def __init__(self, name):
        super().__init__(name, sources=[])


class CMakeBuild(build_ext):
    """Build and configure a CMake extension."""

    def run(self):
        os.system("cd gdf/lib && cmake . && make")


class CleanCommand(Command):
    """Clean up files resulting from compilation except for .so shared objects."""

    CLEAN_FILES = ["build", "dist", "*.egg-info"]
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        for path_spec in self.CLEAN_FILES:
            paths = glob.glob(os.path.normpath(os.path.join(setup_src, path_spec)))
            for path in paths:
                if not str(path).startswith(setup_src):
                    # In case CLEAN_FILES contains an address outside the package
                    raise ValueError("%s is not a path inside %s" % (path, setup_src))
                shutil.rmtree(path)


# From PySCF - ensure the order of build sub-commands:
from distutils.command.build import build

build.sub_commands = [c for c in build.sub_commands if c[0] == "build_ext"] + [
    c for c in build.sub_commands if c[0] != "build_ext"
]


setup(
    packages=find_packages(exclude=["*examples*"]),
    include_package_data=True,
    ext_modules=[CMakeExtension("gdf/lib")],
    cmdclass={
        "build_ext": CMakeBuild,
        "clean": CleanCommand,
    },
    zip_safe=False,
)
