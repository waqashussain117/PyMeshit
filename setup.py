from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import sys
import os
import setuptools

__version__ = '0.1.0'  # Updated version for first public release

class get_pybind_include(object):
    """Helper class to determine the pybind11 include path"""
    def __str__(self):
        import pybind11
        return pybind11.get_include()

ext_modules = [
    Extension(
        'meshit.core._meshit',
        sources=[
            'src/python_bindings_minimal.cpp',
        ],
        include_dirs=[
            'src/',
            get_pybind_include(),
        ],
        language='c++',
        define_macros=[('VERSION_INFO', __version__)],
    ),
]

def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on the specified compiler."""
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True

def cpp_flag(compiler):
    """Return the -std=c++[11/14/17] compiler flag."""
    flags = ['-std=c++17', '-std=c++14', '-std=c++11']
    for flag in flags:
        if has_flag(compiler, flag): return flag
    raise RuntimeError('Unsupported compiler -- at least C++11 support is needed!')

class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc', '/DWIN32', '/D_WINDOWS', '/O2', '/std:c++17'],
        'unix': [],
    }
    l_opts = {
        'msvc': [],
        'unix': [],
    }

    if sys.platform == 'darwin':
        darwin_opts = ['-stdlib=libc++', '-mmacosx-version-min=10.7']
        c_opts['unix'] += darwin_opts
        l_opts['unix'] += darwin_opts

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        link_opts = self.l_opts.get(ct, [])
        
        if ct == 'unix':
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
        elif ct == 'msvc':
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
        
        for ext in self.extensions:
            ext.extra_compile_args = opts
            ext.extra_link_args = link_opts
        
        build_ext.build_extensions(self)

setup(
    name='meshit',
    version=__version__,
    author='Waqas Hussain',
    author_email='waqas.hussain117@gmail.com',
    description='Python library for mesh generation and manipulation with C++ backend',
    long_description=open('README.md', 'r', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/waqashussain/meshit',
    project_urls={
        'Bug Tracker': 'https://github.com/waqashussain/meshit/issues',
        'Documentation': 'https://github.com/waqashussain/meshit#readme',
        'Source Code': 'https://github.com/waqashussain/meshit',
    },
    packages=find_packages(),
    ext_modules=ext_modules,
    install_requires=['pybind11>=2.5.0'],
    python_requires='>=3.7',
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
    # license="AGPL-3.0",  # REMOVE THIS LINE
    keywords='mesh triangulation geometry surface VTU paraview',
    # Keep the classifiers section as is
    include_package_data=True,
)