from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import sys
import os
import setuptools

__version__ = '0.1.0'

# Set environment variables for MinGW
if sys.platform == 'win32':
    os.environ['PATH'] = 'C:\\msys64\\mingw64\\bin;' + os.environ['PATH']

QT_BASE_PATH = 'C:/Qt/6.8.2/mingw_64'
QT_INCLUDE_PATH = os.path.join(QT_BASE_PATH, 'include')
QT_LIB_PATH = os.path.join(QT_BASE_PATH, 'lib')
QT_BIN_PATH = os.path.join(QT_BASE_PATH, 'bin')

class get_pybind_include(object):
    def __str__(self):
        import pybind11
        return pybind11.get_include()

ext_modules = [
    Extension(
        'meshit.core._meshit',
        sources=['src/python_bindings_minimal.cpp'],
        include_dirs=[
            str(get_pybind_include()),
            'src',
            'include',
            QT_INCLUDE_PATH,
            os.path.join(QT_INCLUDE_PATH, 'QtCore'),
            os.path.join(QT_INCLUDE_PATH, 'QtGui'),
            os.path.join(QT_INCLUDE_PATH, 'QtWidgets'),
            os.path.join(QT_INCLUDE_PATH, 'QtOpenGLWidgets'),
        ],
        library_dirs=[
            QT_LIB_PATH,
            QT_BIN_PATH,
        ],
        libraries=[
            'Qt6Core',
            'Qt6Gui',
            'Qt6Widgets',
            'Qt6OpenGLWidgets',
        ],
        language='c++',
        define_macros=[
            ('VERSION_INFO', __version__),
            ('QT_CORE_LIB', '1'),
            ('QT_GUI_LIB', '1'),
            ('QT_WIDGETS_LIB', '1'),
            ('QT_OPENGLWIDGETS_LIB', '1'),
            ('WIN32', '1'),
            ('NOMINMAX', '1'),
            ('NOEXODUS', '1'),
        ],
    )
]

class BuildExt(build_ext):
    def build_extensions(self):
        if self.compiler.compiler_type == 'mingw32':
            for e in self.extensions:
                e.extra_compile_args = [
                    '-O2',
                    '-std=c++17',
                    '-Wall',
                    '-DWIN32',
                    '-D_WINDOWS',
                    '-DQT_NO_DEBUG',
                    '-Wno-unused-variable',  # Suppress unused variable warnings
                    '-Wno-reorder',         # Suppress reorder warnings
                    f'-I{QT_INCLUDE_PATH}',
                    f'-I{os.path.join(QT_INCLUDE_PATH, "QtCore")}',
                    f'-I{os.path.join(QT_INCLUDE_PATH, "QtGui")}',
                    f'-I{os.path.join(QT_INCLUDE_PATH, "QtWidgets")}',
                    f'-I{os.path.join(QT_INCLUDE_PATH, "QtOpenGLWidgets")}',
                ]
                e.extra_link_args = [
                    f'-L{QT_LIB_PATH}',
                    '-lQt6Core',
                    '-lQt6Gui',
                    '-lQt6Widgets',
                    '-lQt6OpenGLWidgets',
                    '-static-libgcc',
                    '-static-libstdc++',
                ]

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
    packages=find_packages(),
    ext_modules=ext_modules,
    install_requires=['pybind11>=2.5.0'],
    python_requires='>=3.7',
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
    include_package_data=True,
)