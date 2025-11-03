from setuptools import setup, Extension, find_packages
import pybind11
import sys

# Platform-specific compilation flags
if sys.platform == 'win32':
    extra_compile_args = ['/std:c++14', '/O2', '/openmp']
    extra_link_args = ['/openmp']
elif sys.platform == 'darwin':
    extra_compile_args = ['-std=c++11', '-O3']
    extra_link_args = []
    print("Warning: OpenMP may not be available on macOS with default compiler")
else:
    extra_compile_args = ['-std=c++11', '-O3', '-fopenmp']
    extra_link_args = ['-fopenmp']

common_args = {
    'include_dirs': [pybind11.get_include()],
    'language': 'c++',
    'extra_compile_args': extra_compile_args,
    'extra_link_args': extra_link_args,
}

ext_modules = [
    Extension(
        'graph.graph_structure_features',  # Matches your compiled .so file
        ['src/struct_features.cpp'],   # Actual source location
        **common_args
    ),
]

setup(
    name='ipag-gin-cfexplainer',
    version='0.1.0',
    description='IPAG Graph Isomorphism Network with Counterfactual Explainer',
    author='Satya',
    python_requires='>=3.7',
    packages=find_packages(),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'torch',
        'transformers',
        'pybind11',
        'pandas',
        'scikit-learn',
        'networkx',
        'typing_extentions',
    ],
    ext_modules=ext_modules,
    zip_safe=False,
)