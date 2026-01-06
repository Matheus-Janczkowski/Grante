# Grante
Repository to store the Grante's Wonderland.

1. Link to the booklet on installation of a miscelaneous of software: https://www.overleaf.com/read/wbxhncmtnmkm#0eb73b

2. Link to the booklet on python programming: https://www.overleaf.com/read/hcmfzzsrhndj#00fdb1

3. Link to the booklet on writing in LaTeX: https://www.overleaf.com/read/sdrvfrpdjhft#66d6f9

# Installation
Download the repository, unzip the file, put it in a suitable place for you.
Activate a python virtual environment (follow instruction in the booklet 1. 
to create a virtual environment if you don't have one), go into the directory 
where the files are located through the virtual environment terminal. Then, 
type in terminal (instead of python you might need to explicitely type in
the version, like python3):

python setup.py bdist_wheel sdist

pip install .

To test the installation:

python

import Grante