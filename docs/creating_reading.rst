Creating and Reading Emission Cubes
===================================

Importing
---------

The :class:`~modspectra.cube.EmissionCube` class is works off the 
:class:`spectral-cube.SpectralCube` class. This wrapper provides methods 
to create, read, write, and view 3D synthetic cubes of HI and H-Alpha.

    >>> from modspectra.cube import EmissionCube

Creating a new cube
-------------------

You can create your own synthetic cube locally and vary the parameters
of the models as you please. The relevant parameters to the models are 
described in Krishnarao, Benjamin & Haffner (2019). You can create the default 
models using the parameter values from this paper using::

    # HI Model of Liszt & Burton (1982) rescaled to model galcen_distance
    >>> hi_cube = EmissionCube.create_LB82()

This cube can be written as a FITS file using::
    
    >>> hi_cube.write("LisztBurton1982_HI_Model.fits", model = True)

The `model = True` keyword saves the associated model parameters in the fits header

Similarly, you can create the H-Alpha model using::

    # H-Alpha model of Krishnarao et al. (2019) including dust reddening
    >>> ha_cube = EmissionCube.create_DK19()


Reading from a file
-------------------

You can read in any fits data cube that usually supported by `spectral-cube`::

    >>> hi_cube = EmissionCube.read("LisztBurton1982_HI_Model.fits")

Using EmissionCube, you can specifically read in models created by modspectra
and still load in parameters of the model. You can load in the HI model and print
some parameters on the tilt angles using the `model = True` keyword::

    >>> hi_cube = EmissionCube.read("LisztBurton1982_HI_Model.fits", model = True)
    >>> print("Tilt Angles of the Liszt & Burton (1982) HI Model are:")
    >>> print("Alpha = {0:.2f}, Beta = {1:.2f}, Theta = {2:.2f}".format(hi_cube.alpha, 
                                                                        hi_cube.beta, 
                                                                        hi_cube.theta))
    Tilt Angles of the Liszt & Burton (1982) HI Model are:
    Alpha = 13.50 deg, Beta = 20.00 deg, Theta = 48.50 deg

Modifying Parameters
--------------------

You can modify any of the model parameters via keywords to 
`create_DK19` and `create_LB82`. A full list of the keywords can be found in the 
:class:`~modspectra.cube.EmissionCube` class page::

    # Compute H-Alpha cube with no reddening to see what it would appear like 
    # in a Galaxy with no dust
    resolution = (128,128,128)
    >>> ha_cube = EmissionCube.create_DK19(resolution = resolution, redden = False)

    # Compute HI cube with larger scale height and higher resolution
    # Use memory mapping via dask
    resolution = (400,400,400)
    Hz = 0.2 # kpc
    >>> hi_cube = EmissionCube.create_LB82(resolution = resolution, Hz = Hz,  
                                           memmap = True)



