ModSpectra Documentation
========================

The modspectra package provides an easy way to create synthetic 3D data cubes 
of HI and H-Alpha spectra in the Galaxy. This package uses the ``spectral-cube`` 
package to handle the data cube and the primary class, ``EmissionCube``, is 
built around this. The current main features of modspectra are to complement
the work of Krishnarao, Benjamin, Haffner (2019) on a model of a Tilted 
Elliptical Gas Disk around Galactic Center. 
It provides the following main features:

-The ability to create synthetic 3D Data cubes of HI emission following the 
model of Liszt & Burton (1982)
-The ability to create synthetic 3D Data cubes of H-Alpha emission following
the model of Krishnarao, Benjamin, Haffner (2019), including reddening by dust.
-The ability to create synthetic line spectra for any provided coordinate in
the sky for HI or H-Alpha emission from the models mentioned above.

Quick Start
-----------

Here is a simple script demonstrating the modspectra package:

	>>> from modspectra.cube import EmissionCube
	>>> hi_cube = EmissionCube.create_LB82() # Create HI Model Cube
	>>> ha_cube = EmissionCube.create_DK19() # Create H-Alpha Model Cube

	# Save Data Cubes as fits files
	>>> hi_cube.write("LisztBurton1982_HI_Model.fits", model = True)
	>>> ha_cube.write("DK2019_HA_Model.fits", model = True)

Using ModSpectra
----------------

This package centers around using the `spectral cube`_ package for handling
the 3D data cubes created. Please refer to their documentation.

.. _spectral cube: https://spectral-cube.readthedocs.io/en/latest/#


Getting started
^^^^^^^^^^^^^^^

.. toctree::
  :maxdepth: 1

  installing.rst
  creating_reading.rst
  TiltedDisk_coordinates.rst
  ppp_cubes.rst
  plotting_moments.rst
  custom_ppv_cubes.rst