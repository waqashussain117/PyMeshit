cmake_minimum_required(VERSION 3.15)
project(meshit)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python and pybind11
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Find Qt
set(CMAKE_PREFIX_PATH "C:/Qt/6.8.2/mingw_64")
find_package(Qt6 COMPONENTS Core Gui Widgets OpenGLWidgets REQUIRED)

# Add the extension module
pybind11_add_module(_meshit src/python_bindings_minimal.cpp)

# Add include directories
target_include_directories(_meshit PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    "C:/Qt/Tools/QtCreator/share/qtcreator/cplusplus/wrappedQtHeaders"
    "C:/Qt/Tools/QtCreator/share/qtcreator/indexer_preincludes"
)

# Link Qt libraries
target_link_libraries(_meshit PRIVATE
    Qt6::Core
    Qt6::Gui
    Qt6::Widgets
    Qt6::OpenGLWidgets
)

# Add compile definitions
target_compile_definitions(_meshit PRIVATE
    VERSION_INFO="0.1.1"
    QT_CORE_LIB=1
    QT_GUI_LIB=1
    QT_WIDGETS_LIB=1
    QT_OPENGLWIDGETS_LIB=1
    WIN32=1
    NOMINMAX=1
    NOEXODUS=1
)

# Install the module
install(TARGETS _meshit DESTINATION meshit/core) 