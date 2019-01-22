import pytest
from numpy.random import randn
from numpy.random import random
import numpy as np


def test_bd_solver_along_major_axis():
    from ..cube import bd_solver
    from numpy import zeros, allclose
    '''
    Test case is maximal ellipse size from Liszt & Burton (1980)
    ad = ad_max = 3.1 * bd_max = 3.1 * 0.6 = 1.86 = x_coord
    y_coord = 0.
    z_coord = rand between -0.2 and 0.2
    '''
    xyz = zeros((3,1))
    xyz[2,0] = randn()*0.1
    z_sigma_lim = 5. #testing at z = 0
    Hz = 0.1
    bd_max = random()
    el_constant1 = random() + 1.
    el_constant2 = random() + 1.
    xyz[0,0] = (el_constant1 + el_constant2) * bd_max
    test_res = bd_solver(0, xyz, z_sigma_lim, Hz, bd_max, el_constant1, el_constant2)
    assert allclose(bd_max, test_res)

def test_ellipse_equation():
    from ..cube import ellipse_equation 
    '''
    Test case is maximal ellipse size from Liszt & Burton (1980)
    '''
    bd = 0.6
    el_constant1 = 1.6
    el_constant2 = 1.5
    bd_max = 0.6
    x_coord = 0.0
    y_coord = 0.6

    assert ellipse_equation(bd, el_constant1, el_constant2, bd_max, x_coord, y_coord) == 0.0

def test_memmap_density():
    from ..cube import EllipticalLBD
    from numpy import allclose
    '''
    Using Default Liszt & Burton (1980) parameters as test case - 
    parameters don't really matter here at all, so parameters are sometimes random
    '''
    resolution = (64,64,64)
    bd_max = random()
    Hz = random()
    z_sigma_lim = 3
    dens0 = random()
    velocity_factor = 100*random()
    vel_0 = 350*random()
    el_constant1 = 1.0 + random()
    el_constant2 = 1.0 + random()
    alpha = 23. + 5. * randn()
    beta = 22. + 5. * randn()
    theta = 48. + 5 * randn()
    L_range = [-10,10]
    B_range = [-8,8]
    D_range = [6,10]
    success = False
    while not success:
        try:
            res_no_memmap = EllipticalLBD(resolution, bd_max, Hz, z_sigma_lim, dens0, 
                velocity_factor, vel_0, el_constant1, el_constant2, alpha, beta, theta,
                 L_range, B_range, D_range)
        except ValueError:
            bd_max = random()
            success = False
        else:
            success = True
    res_memmap = EllipticalLBD(resolution, bd_max, Hz, z_sigma_lim, dens0, 
        velocity_factor, vel_0, el_constant1, el_constant2, alpha, beta, theta,
         L_range, B_range, D_range, memmap = True, da_chunks_xyz = 32)
    assert allclose(res_memmap[1],res_no_memmap[1])

def test_non_detection():
    from ..cube import EmissionCube
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from numpy import allclose
    '''
    Test that an anti-center pointing returns zero emission
    '''
    l = 180. + randn()*130.
    b = 0. + randn()*20.
    c = SkyCoord(l = l*u.deg, b = b*u.deg, frame = 'galactic', galcen_distance = 8.127*u.kpc)
    spec = EmissionCube.create_DK19_spectrum(c, 0.5 * u.deg, redden = False)
    assert allclose(spec.value, np.zeros_like(spec.value))








