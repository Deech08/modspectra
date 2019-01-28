import pytest
from numpy.random import randn
from numpy.random import random
import numpy as np

def test_non_detection():
    from ..cube import EmissionCube
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    '''
    Test that an anti-center pointing returns zero emission
    '''
    l = 180. + randn()*130.
    b = 0. + randn()*20.
    c = SkyCoord(l = l*u.deg, b = b*u.deg, frame = 'galactic', galcen_distance = 8.127*u.kpc)
    spec = EmissionCube.create_DK19_spectrum(c, 0.5 * u.deg, redden = False)
    assert np.allclose(spec.value, np.zeros_like(spec.value))

def test_coordinate_error():
    from ..cube import EmissionCube
    import astropy.units as u
    '''
    Ensure that a SkyCoord Object is required
    '''
    l = 0. + randn()*5.
    b = 0. + randn()*3.
    try:
        spec = EmissionCube.create_DK19_spectrum((l,b), 0.5 * u.deg, redden = False)
    except ValueError:
        assert True
    else:
        assert False

def test_galcen_distance():
    from ..cube import EmissionCube
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    '''
    Ensure that a default galcen_distnace is adopted
    '''
    l = 0. + randn()*5.
    b = 0. + randn()*3.
    c = SkyCoord(l = l*u.deg, b = b*u.deg, frame = 'galactic')
    c2 = SkyCoord(l = l*u.deg, b = b*u.deg, frame = 'galactic', galcen_distance = 8.127*u.kpc)
    spec = EmissionCube.create_DK19_spectrum(c, 0.5 * u.deg, redden = False)
    spec2 = EmissionCube.create_DK19_spectrum(c2, 0.5 * u.deg, redden = False)
    assert np.allclose(spec.value, spec2.value)

def test_radius_degrees():
    from ..cube import EmissionCube
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    '''
    Ensure that a default units for radius are in
    '''
    l = 0. + randn()*5.
    b = 0. + randn()*3.
    c = SkyCoord(l = l*u.deg, b = b*u.deg, frame = 'galactic', galcen_distance = 8.127*u.kpc)
    r1 = np.abs( randn()*1000.) * u.arcmin
    r2 = r1.to(u.deg).value
    spec = EmissionCube.create_DK19_spectrum(c, r1, redden = False)
    spec2 = EmissionCube.create_DK19_spectrum(c, r2, redden = False)
    assert np.allclose(spec.value, spec2.value)
    
def test_reduce_cube():
    from ..cube import EmissionCube
    import astropy.units as u
    '''
    Ensure that reduce cube returns same result as not using it with lon/lat
    '''
    cube = EmissionCube.create_LB82(resolution = (64,64,64), L_range = [-5,5], B_range = [-5,5])
    l = randn()*3.
    b = randn()*3.
    while (np.abs(l) > 4) | (np.abs(b) > 4):
        l = randn()*3.
        b = randn()*3.
    radius = 1.5 * u.deg
    spectrum_reduce = cube.extract_beam(longitude = l, latitude = b, 
                                        reduce_cube = True, radius = radius)
    spectrum = cube.extract_beam(longitude = l*u.deg, latitude = b*u.deg, 
                                 reduce_cube = False, radius = radius)
    assert np.allclose(spectrum_reduce.value, spectrum.value)

def test_coord_lon_lat():
    from ..cube import EmissionCube
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    '''
    Ensure that using SKyCoord or specifiying lon/lat return same result
    '''
    cube = EmissionCube.create_LB82(resolution = (64,64,64), L_range = [-5,5], B_range = [-5,5])
    l = randn()*3.
    b = randn()*3.
    while (np.abs(l) > 4) | (np.abs(b) > 4):
        l = randn()*3.
        b = randn()*3.
    radius = 1.5 * u.deg
    c = SkyCoord(l = l*u.deg, b = b*u.deg, frame = 'galactic', galcen_distance = 8.127*u.kpc)
    spec = cube.extract_beam(longitude = l, latitude = b, radius = radius)
    spec2 = cube.extract_beam(coordinate = c, radius = radius)
    assert np.allclose(spec.value, spec2.value)

def test_coord_reduce():
    from ..cube import EmissionCube
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    '''
    Ensure that using reduce cube with coordinate works
    '''
    cube = EmissionCube.create_LB82(resolution = (64,64,64), L_range = [-5,5], B_range = [-5,5])
    l = randn()*3.
    b = randn()*3.
    while (np.abs(l) > 4) | (np.abs(b) > 4):
        l = randn()*3.
        b = randn()*3.
    radius = 1.5 * u.deg
    c = SkyCoord(l = l*u.deg, b = b*u.deg, frame = 'galactic', galcen_distance = 8.127*u.kpc)
    spec = cube.extract_beam(coordinate = c, reduce_cube = True, radius = radius)
    spec2 = cube.extract_beam(coordinate = c, reduce_cube = True, radius = radius)
    assert np.allclose(spec.value, spec2.value)

    
    
