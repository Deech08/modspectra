import pytest
from numpy.random import randn
from numpy.random import random
import numpy as np

from ..cube import EmissionCube
import astropy.units as u
cube = EmissionCube.create_LB82(resolution = (64,64,64), L_range = [-5,5]*u.deg, B_range = [-5,5]*u.deg)



def test_reduce_cube():
    '''
    Ensure that reduce cube returns same result as not using it with lon/lat
    '''
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
    from astropy.coordinates import SkyCoord
    '''
    Ensure that using SKyCoord or specifiying lon/lat return same result
    '''
    l = randn()*3.
    b = randn()*3.
    while (np.abs(l) > 4) | (np.abs(b) > 4):
        l = randn()*3.
        b = randn()*3.
    radius = 1.5 * u.deg
    c = SkyCoord(l = l*u.deg, b = b*u.deg, frame = 'galactic', galcen_distance = 8.127*u.kpc)
    spec = cube.extract_beam(longitude = l, latitude = b, radius = radius)
    spec2 = cube.extract_beam(coordinate = c, radius = radius.value)
    assert np.allclose(spec.value, spec2.value)

def test_coord_reduce():
    from astropy.coordinates import SkyCoord
    '''
    Ensure that using reduce cube with coordinate works
    '''
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

def test_extract_spec():
    '''
    Test that the extract_spec method works
    '''
    l = randn()*3.
    b = randn()*3.
    while (np.abs(l) > 4) | (np.abs(b) > 4):
        l = randn()*3.
        b = randn()*3.
    _, spec = cube.extract_spectrum(l, b)
    _, spec2 = cube.extract_spectrum(l*u.deg, b*u.deg)
    assert np.allclose(spec.value, spec2.value)

def test_ad_equation():
    '''
    Test ad equation works
    '''
    bd = random() * 5.
    ad = cube.ad(bd)
    ad2 = cube.ad(bd*u.kpc)
    assert np.allclose(ad.value, ad2.value)
