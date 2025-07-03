from setuptools import setup, Extension  
from Cython.Build import cythonize
import Cython.Compiler.Options
import numpy  



extensions = [
    Extension(
        "hybrid_A_star_TF_Dec_13",
        ["hybrid_A_star_TF_Dec_13.pyx"],
        include_dirs=[numpy.get_include()] , ##extra_compile_args=["-O2"],  # Use optimization level 2
        #language="c++"
        )
] 

setup(
    ext_modules=cythonize(extensions),
)
