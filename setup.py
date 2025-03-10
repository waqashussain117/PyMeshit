from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import sys
import os
import setuptools

__version__ = '0.1.1'

QT_BASE_PATH = 'C:/Qt/6.8.2/msvc2022_64'
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
            ('_SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING', '1'),
        ],
    )
]

class BuildExt(build_ext):
    def build_extensions(self):
        opts = [
            '/O2',
            '/std:c++17',
            '/W3',
            '/DWIN32',
            '/D_WINDOWS',
            '/DQT_NO_DEBUG',
            '/EHsc',
            '/MD',
            '/Zc:__cplusplus',
            '/permissive-',
            '/Zc:preprocessor',
            '/Zc:externConstexpr',
            '/Zc:throwingNew',
            '/utf-8',
            '/bigobj',
        ]
        
        # Add Qt include paths
        qt_includes = [
            QT_INCLUDE_PATH,
            os.path.join(QT_INCLUDE_PATH, 'QtCore'),
            os.path.join(QT_INCLUDE_PATH, 'QtGui'),
            os.path.join(QT_INCLUDE_PATH, 'QtWidgets'),
            os.path.join(QT_INCLUDE_PATH, 'QtOpenGLWidgets'),
        ]

        for ext in self.extensions:
            ext.extra_compile_args = opts
            for inc in qt_includes:
                ext.extra_compile_args.append(f'/I{inc}')

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
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)