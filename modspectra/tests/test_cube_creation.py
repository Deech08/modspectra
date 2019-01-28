import pytest
from numpy.random import randn
from numpy.random import random
import numpy as np
# Set up the random number generator.
np.random.seed(1234)


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

def test_memmap_cube():
    from ..cube import EmissionCube
    from numpy import allclose
    '''
    Ensure memmap and non memmap versions of final Emission Cube are the same
    '''
    resolution = (32,32,32)
    memmap_cube = EmissionCube.create_DK19(memmap = True, resolution = resolution, redden = False)
    cube = EmissionCube.create_DK19(memmap = False, resolution = resolution, redden = False)
    assert allclose(memmap_cube, cube)

# Redden will not work for now
# Can't figure out how to fetch the Marshall map on travis
##### Note to self / call for help: Is there a simple way to download the data file then 
##### use it for the test alone. Trying to avoid attaching the data file to the package...

# Test will pass if run locally
# def test_memmap_cube_redden():
#     from ..cube import EmissionCube
#     from numpy import allclose
#     '''
#     Ensure memmap and non memmap versions of final Emission Cube are the same with reddening
#     '''
#     resolution = (32,32,32)
#     memmap_cube = EmissionCube.create_DK19(memmap = True, resolution = resolution, redden = True)
#     cube = EmissionCube.create_DK19(memmap = False, resolution = resolution, redden = True)
#     assert allclose(memmap_cube, cube)

def test_memmap_cube_flaring_radial_false():
    from ..cube import EmissionCube
    from numpy import allclose
    '''
    Ensure memmap and non memmap versions of final Emission Cube are the same with parameter
    flaring_radial = False
    '''
    resolution = (32,32,32)
    memmap_cube = EmissionCube.create_DK19(memmap = True, resolution = resolution, 
        redden = False, flaring_radial = False)
    cube = EmissionCube.create_DK19(memmap = False, resolution = resolution, 
        redden = False, flaring_radial = False)
    assert allclose(memmap_cube, cube)

def test_memmap_cube_hi():
    from ..cube import EmissionCube
    from numpy import allclose
    '''
    Ensure memmap and non memmap versions of final Emission Cube are the same
    '''
    resolution = (32,32,32)
    memmap_cube = EmissionCube.create_LB82(memmap = True, resolution = resolution)
    cube = EmissionCube.create_LB82(memmap = False, resolution = resolution)
    assert allclose(memmap_cube, cube)

def test_return_all():
    from ..cube import EmissionCube
    '''
    Ensure return_all functionality works by checking output keys in cube
    '''
    resolution = (32,32,32)
    memmap_cube = EmissionCube.create_LB82(memmap = True, resolution = resolution, return_all = True)
    cube = EmissionCube.create_LB82(memmap = False, resolution = resolution, return_all = True)
    assert memmap_cube.LBD_output_keys == cube.LBD_output_keys

def test_min_bd():
    from ..cube import EmissionCube
    from numpy import allclose
    '''
    Ensure min_bd works to create a ring
    '''
    resolution = (32,32,32)
    min_bd = random() * 0.3
    memmap_cube = EmissionCube.create_LB82(memmap = True, resolution = resolution, min_bd = min_bd)
    cube = EmissionCube.create_LB82(memmap = False, resolution = resolution, min_bd = min_bd)
    assert allclose(memmap_cube, cube)

def test_unit_ranges():
    from ..cube import EmissionCube
    from numpy import allclose
    import astropy.units as u
    '''
    Ensure different units input as cube ranges still work
    '''
    resolution = (32,32,32)
    L_range = [-10,10] * u.deg
    B_range = [-8,8] * u.deg
    D_range = [5,10] * u.kpc
    cube = EmissionCube.create_LB82(resolution = resolution, 
                                    L_range = L_range, 
                                    B_range = B_range, 
                                    D_range = D_range)
    cube_dif_unit = EmissionCube.create_LB82(resolution = resolution, 
                                             L_range = L_range.to(u.arcmin), 
                                             B_range = B_range.to(u.arcsec), 
                                             D_range = D_range.to(u.pc))
    assert allclose(cube, cube_dif_unit)

def test_angle_units():
    from ..cube import EmissionCube
    from numpy import allclose
    import astropy.units as u
    '''
    Ensure different units input as cube ranges still work
    '''
    resolution = (32,32,32)
    alpha = 13.5 * u.deg
    beta = 20. * u.deg
    theta = 48.5 * u.deg
    cube = EmissionCube.create_LB82(resolution = resolution, 
                                    alpha = alpha.value, 
                                    beta = beta.value, 
                                    theta = theta.value)
    cube_dif_unit = EmissionCube.create_LB82(resolution = resolution, 
                                             alpha = alpha.to(u.arcmin), 
                                             beta = beta.to(u.arcsec), 
                                             theta = theta.to(u.rad))
    assert allclose(cube, cube_dif_unit)

def test_caseA():
    from ..cube import EmissionCube
    from numpy import allclose
    '''
    Ensure memmap and non memmap versions of final Emission Cube are the same with case A 
    '''
    resolution = (32,32,32)
    memmap_cube = EmissionCube.create_DK19(memmap = True, resolution = resolution, 
                                            redden = False, case = 'A')
    cube = EmissionCube.create_DK19(memmap = False, resolution = resolution, 
                                    redden = False, case = 'A')
    assert allclose(memmap_cube, cube)

def test_parameter_units():
    from ..cube import EmissionCube
    from numpy import allclose
    import astropy.units as u
    '''
    Ensure default units and conversions are made for other parameters of model
    '''
    resolution = (32,32,32)
    # Set parameters
    bd_max = 0.2 + random() * 0.3
    bd_max *= u.kpc
    Hz = 0.05 + random() * 0.2
    Hz *= u.kpc
    dens0 = 0.2 + random() * 0.2
    dens0 *= (u.cm)**-3
    vel_0 = 350 + random() * 20.
    vel_0 *= u.km/u.s
    vel_disp = 7 + random() * 3.
    vel_disp *= u.km/u.s
    vmin = -340 - random() * 20.
    vmin *= u.km/u.s
    vmax = 340 + random() * 20.
    vmax *= u.km/u.s

    vel_resolution = 200

    cube = EmissionCube.create_LB82(resolution = resolution, 
                                   bd_max = bd_max.value, 
                                   Hz = Hz.value, 
                                   dens0 = dens0.value, 
                                   vel_0 = vel_0.value, 
                                   vel_disp = vel_disp.value, 
                                   vmin = vmin.value, 
                                   vmax = vmax.value, 
                                   vel_resolution = vel_resolution)
    cube_dif_unit = EmissionCube.create_LB82(resolution = resolution,
                                             bd_max = bd_max.to(u.km), 
                                             Hz = Hz.to(u.pc), 
                                             dens0 = dens0.to((u.m)**-3), 
                                             vel_0 = vel_0.to(u.m/u.s), 
                                             vel_disp = vel_disp.to(u.kpc/u.Gyr), 
                                             vmin = vmin.to(u.cm/u.s), 
                                             vmax = vmax.to(u.km/u.yr), 
                                             vel_resolution = vel_resolution)
    assert allclose(cube, cube_dif_unit)

