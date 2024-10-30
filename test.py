import sys
import site

# Check installed packages path
print(site.getsitepackages())

# Or check if a specific package (e.g., numpy) is installed in this environment
try:
    import numpy
    print("Numpy version:", numpy.__version__)
except ImportError:
    print("Numpy is not installed in this environment.")
    