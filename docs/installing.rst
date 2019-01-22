Installing ``modspectra``
============================

Requirements
------------

This package has the following dependencies:

* `Python <http://www.python.org>`_ 3.6 or later
* `Numpy <http://www.numpy.org>`_ 1.8 or later
* `Astropy <http://www.astropy.org>`__ 1.0 or later
* `Scipy <https://www.scipy.org/>`_ 1.2 or later
* `spectral cube <https://spectral-cube.readthedocs.io/en/latest/#>`_ >=0.4.4
* `numexpr <https://numexpr.readthedocs.io/en/latest/user_guide.html>`_ 2.0 or later
* `extinction <https://extinction.readthedocs.io/en/latest/>`_>=0.4.0
* `dustmaps <https://github.com/gregreen/dustmaps>`_>=1.0.3
* `dask <https://dask.org/>`_ optional
(Used when creating high resolution cubes and memmap = True is set)
* `Regions <https://astropy-regions.readthedocs.io/en/latest>`_ >=0.3dev, optional
  (Serialises/Deserialises DS9/CRTF region files and handles them. Used when
  extracting a subcube from region)

Installation
------------

To install the latest developer version of modspectra you can type::

    git clone https://github.com/Deech08/modspectra.git
    cd modspectra
    python setup.py install

You may need to add the ``--user`` option to the last line `if you do not
have root access <https://docs.python.org/2/install/#alternate-installation-the-user-scheme>`_.
You can also install the latest developer version in a single line with pip::

    pip install git+https://github.com/Deech08/modspectra.git


