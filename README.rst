ModSpectra Documentation
========================

.. image:: http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat
    :target: http://www.astropy.org
    :alt: Powered by Astropy Badge

.. image:: https://readthedocs.org/projects/modspectra/badge/?version=latest
	:target: https://modspectra.readthedocs.io/en/latest/?badge=latest
	:alt: Documentation Status

.. image:: https://travis-ci.org/Deech08/modspectra.svg?branch=master
    :target: https://travis-ci.org/Deech08/modspectra

.. image:: https://coveralls.io/repos/github/Deech08/modspectra/badge.svg?branch=master&service=github
	:target: https://coveralls.io/github/Deech08/modspectra?branch=master


The modspectra package provides an easy way to create synthetic 3D data cubes 
of HI and H-Alpha spectra in the Galaxy. This package uses the ``spectral-cube`` 
package to handle the data cube and the primary class, ``EmissionCube``, is 
built around this. The current main features of modspectra are to complement
the work of Krishnarao, Benjamin, Haffner (2019) on a model of a Tilted 
Elliptical Gas Disk around Galactic Center. 
It provides the following main features:

-The ability to create synthetic 3D Data cubes of HI emission following the 
model of Liszt & Burton (1980)
-The ability to create synthetic 3D Data cubes of H-Alpha emission following
the model of Krishnarao, Benjamin, Haffner (2020), including reddening by dust.
-The ability to create synthetic line spectra for any provided coordinate in
the sky for HI or H-Alpha emission from the models mentioned above.


License
-------

This project is Copyright (c) Dhanesh Krishnarao (DK) and licensed under
the terms of the BSD 3-Clause license. This package is based upon
the [Astropy package template](https://github.com/astropy/package-template)
which is licensed under the BSD 3-clause licence. See the licenses folder for
more information.



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

Quick Start
-----------

Here is a simple script demonstrating the modspectra package::

	>>> from modspectra.cube import EmissionCube
	>>> hi_cube = EmissionCube.create_LB80() # Create HI Model Cube
	>>> ha_cube = EmissionCube.create_DK20() # Create H-Alpha Model Cube

	# Save Data Cubes as fits files
	>>> hi_cube.write("LisztBurton1980_HI_Model.fits", model = True)
	>>> ha_cube.write("DK2020_HA_Model.fits", model = True)
