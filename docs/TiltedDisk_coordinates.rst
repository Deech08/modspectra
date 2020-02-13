TiltedDisk Coordinate Frame 
===========================

The :class:`~modspectra.cube.TiltedDisk` class is an astropy coordinate frame 
that describes the tilted elliptical disk structure of Liszt & Burton (1982) 
as modified in Krishnarao, Benjamin, & Haffner (2020). Coordinate transformations
can be done easily using the standard astropy.coordinates methods and the class
is compatible with the `astropy.coordinates.SkyCoord` object::

    from modspectra.cube import TiltedDisk
    from astropy.coordinates import SkyCoord
    import astropy.units as u

By default, `TiltedDisk` assumes the tilt angles used in Krishnarao, Benjamin, 
& Haffner (2019), but these can be manually set as well::

    c = SkyCoord(x = 1.5*u.kpc, y = .3 * u.kpc, z = -.5 *u.kpc, 
                 frame = TiltedDisk(alpha = 13.5*u.deg, 
                                    beta = 20*u.deg, 
                                    theta = 48.5*u.deg), 
                 galcen_distance = 8.127*u.kpc)

    c_gal = c.transform_to('galactic')

    print(c_gal)
    <SkyCoord (Galactic): (l, b, distance) in (deg, deg, kpc)
        ( 354.48642214,  1.01834443,  9.48797151)>


