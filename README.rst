Package to create 3D synthetic spectroscopic observations of neutral and ionized gas.
-------------------------------------------------------------------------------------

.. image:: http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat
    :target: http://www.astropy.org
    :alt: Powered by Astropy Badge




License
-------

This project is Copyright (c) Dhanesh Krishnarao (DK) and licensed under
the terms of the BSD 3-Clause license. This package is based upon
the [Astropy package template](https://github.com/astropy/package-template)
which is licensed under the BSD 3-clause licence. See the licenses folder for
more information.



Installation
------------

To install modspectra, download the package using::
	
	$ git clone https://github.com/Deech08/modspectra.git

Install using pip::
	
	pip install -e /path/to/package

Quick Start
-----------

To recreate a 3D fits cube of HI-21cm associated with the Tilted Elliptical 
Disk of Liszt & Burton (1982)::
	
	from modspectra.cube import EmissionCube
	hi_model = EmissionCube.create_LB82()
	hi_model.write("LisztBurton1982_HI_Model.fits", format = 'fits', model = True)

To recreate a 3D fits cube of H-Alpha associated with the Tilted Elliptical
Disk of Krishnarao, Benjamin & Haffner (2019)::
	from modspectra.cube import EmissionCube
	ha_model = EmissionCube.create_DK19()
	ha_model.write("DK2019_HA_Model.fits", format = 'fits', model = True)
