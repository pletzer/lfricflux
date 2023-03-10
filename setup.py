from distutils.core import setup
import re

PACKAGE = 'lfricflux'

with open("version.txt") as f:
    VERSION = f.read().strip()

# generate {PACKAGE}/__init__.py from {PACKAGE}/__init__.py.in
init_file_content = "" 
with open(f"{PACKAGE}/__init__.py.in") as fi:
    init_file_content = re.sub(r"@VERSION@", VERSION, fi.read())
    with open(f"{PACKAGE}/__init__.py", "w") as fo:
        fo.write(init_file_content)

setup(name='lfricflux',
)
