# Using file formatting based on dfm/emcee 
#!/bin/bash -x

# http://conda.pydata.org/docs/travis.html#the-travis-yml-file
if [[ "$TRAVIS_OS_NAME" == "osx" ]]; then
  wget https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O miniconda.sh;
else
  wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
fi
bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"

# Conda Python
hash -r
conda config --set always_yes yes --set changeps1 no
conda update -q conda
conda info -a
conda create --yes -n test python=$PYTHON_VERSION
source activate test
conda install -q numpy=$NUMPY_VERSION scipy setuptools pytest pytest-cov pip sphinx matplotlib numexpr dask
# conda install -c conda-forge extinction
pip install coveralls
pip install astropy
pip install extinction
if [[ "$TRAVIS_OS_NAME" == "osx" ]]; then
	pip install healpy==1.12.5
else
	pip install healpy
fi
pip install dustmaps
pip install spectral-cube==0.4.3
pip install --no-deps pyregion
pip install pytest-mpl

# Build the extension
python setup.py develop