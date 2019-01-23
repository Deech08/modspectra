import pytest
from numpy.random import randn
from numpy.random import random
import numpy as np

def test_read_write_model_false():
    '''
    Test writing and reading a spectral-cube
    '''
    from ..cube import EmissionCube

    c = EmissionCube.create_LB82(resolution = (32,32,32))
    try:
        c.write("test_io_cube.fits", model = False, overwrite = False)
    except OSError:
        # Need to overwrite test file
        c.write("test_io_cube.fits", model = False, overwrite = True)
    c2 = EmissionCube.read("test_io_cube.fits", model = False)
    assert np.allclose(c,c2)

def test_read_write_model_true():
    '''
    Test writing and reading a spectral-cube
    '''
    from ..cube import EmissionCube

    c = EmissionCube.create_LB82(resolution = (32,32,32))
    try:
        c.write("test_io_cube.fits", model = True, overwrite = False)
    except OSError:
        # Need to overwrite test file
        c.write("test_io_cube.fits", model = True, overwrite = True)
    c2 = EmissionCube.read("test_io_cube.fits", model = True)
    assert np.allclose(c,c2)

def test_read_write_model_parameters():
    '''
    Test writing and reading a spectral-cube
    '''
    from ..cube import EmissionCube

    c = EmissionCube.create_LB82(resolution = (32,32,32))
    try:
        c.write("test_io_cube.fits", model = True, overwrite = False)
    except OSError:
        # Need to overwrite test file
        c.write("test_io_cube.fits", model = True, overwrite = True)
    c2 = EmissionCube.read("test_io_cube.fits", model = True)
    assert c.bd_max == c2.bd_max
