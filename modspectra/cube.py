import numpy as np
import os
from astropy import units as u
from spectral_cube import SpectralCube
import numexpr as ne
import dask.array as da
from dask.diagnostics import ProgressBar
import logging


from astropy.coordinates.representation import CylindricalRepresentation, CartesianRepresentation, CartesianDifferential
import astropy.coordinates as coord
from astropy.coordinates import frame_transform_graph

from astropy import wcs

import scipy.interpolate
import scipy.integrate as integrate

import multiprocessing
from functools import partial

from .cubeMixin import EmissionCubeMixin

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def find_nearest_idx(array,value):
    idx = (np.abs(array-value)).argmin()
    return [idx]

def find_nannearest_idx(array,value):
    idx = np.nanargmin(np.abs(array-value))
    return [idx]

class TiltedDisk(coord.BaseCoordinateFrame):
    """
    A cartesian coordinate system in the frame of the tilted elliptical disk 

    Parameters
    ----------
    representation : `BaseRepresentation` or None
        A representation object or None to have no data (or use the other keywords)

    x : `~astropy.units.Quantity`, optional, must be keyword
        The x coordinate in the tilted disk coordinate system
    y : `~astropy.units.Quantity`, optional, must be keyword
        The y cooridnate in the titled disk coordinate system
    z : `~astropy.units.Quantity`, optional, must be keyword
        The z coordinate in the tilted disk coordinate system
        
    v_x : :class:`~astropy.units.Quantity`, optional, must be keyword
        The x component of the velocity
    v_y : :class:`~astropy.units.Quantity`, optional, must be keyword
        The y component of the velocity
    v_z : :class:`~astropy.units.Quantity`, optional, must be keyword
        The z component of the velocity
    """
    default_representation = coord.CartesianRepresentation
    default_differential = coord.CartesianDifferential

    frame_specific_representation_info = {
        coord.representation.CartesianDifferential: [
            coord.RepresentationMapping('d_x', 'v_x', u.km/u.s),
            coord.RepresentationMapping('d_y', 'v_y', u.km/u.s),
            coord.RepresentationMapping('d_z', 'v_z', u.km/u.s),
        ],
    }

    # Specify frame attributes required to fully specify the frame
    # Rotation angles
    alpha = coord.QuantityAttribute(default=0.*u.rad, unit = u.rad)
    beta = coord.QuantityAttribute(default=0.*u.rad, unit = u.rad)
    theta = coord.QuantityAttribute(default=0.*u.rad, unit = u.rad)



def get_transformation_matrix(tilteddisk_frame, inverse = False):
    """
    Create coordinate transformation matrix for converting between the TiltedDisk frame and Galactocentric frame

    Parameters
    ----------
    tilteddisk_frame: TiltedDisk class Coordinate frame

    inverse: 'bool', optional, must be keyword
    	if True, return the transposed matrix for converting from the Galactocentric frame to the TiltedDisk frame 

    """
    alpha = tilteddisk_frame.alpha.value
    beta = tilteddisk_frame.beta.value
    theta = tilteddisk_frame.theta.value
    # Generate rotation matrix for coordinate transformation into coord.Galactocentric
    R_matrix = np.array([np.cos(beta)*np.cos(theta), np.cos(beta)*np.sin(theta), -np.sin(beta), 
                        -np.cos(theta)*np.sin(alpha)*-np.sin(beta) - np.cos(alpha)*np.sin(theta), 
                        np.cos(alpha)*np.cos(theta) + np.sin(alpha)*np.sin(beta)*np.sin(theta), 
                        np.cos(beta)*np.sin(alpha), 
                        np.cos(alpha)*np.cos(theta)*np.sin(beta) + np.sin(alpha)*np.sin(theta), 
                        -np.cos(theta)*np.sin(alpha) + np.cos(alpha)*np.sin(beta)*np.sin(theta), 
                        np.cos(alpha)*np.cos(beta)]).reshape(3,3)
    if inverse:
        return R_matrix.transpose()
    else:
        return R_matrix
    
@frame_transform_graph.transform(coord.DynamicMatrixTransform, TiltedDisk, coord.Galactocentric)
def td_to_galactocentric(tilteddisk_coord, galactocentric_frame):
    """ Compute the transformation matrix from the Tilted Disk 
        coordinates to Galactocentric coordinates.
    """
    return get_transformation_matrix(tilteddisk_coord)
    
@frame_transform_graph.transform(coord.DynamicMatrixTransform, coord.Galactocentric, TiltedDisk)
def galactocentric_to_td(galactocentric_coord, tilteddisk_frame):
    """ Compute the transformation matrix from Galactocentric coordinates to
        Tilted Disk coordinates.
    """
    return get_transformation_matrix(tilteddisk_frame, inverse = True)



def ellipse_equation(bd, el_constant1, el_constant2, bd_max, x_coord, y_coord):
    """
    Equation for an ellipse in the form from Burton & Liszt (1978)
    Function serves to be used in scipy.optimize.brenth to solve for bd 

    Parameters
    ----------
    bd: 'number'
    	semi-minor axis of ellipse
    el_constant1: 'number'
    	First parameter for defining ellipse 
    el_constant2: 'number'
    	second parameter for defining ellipse
    bd_max: 'number'
    	Maximum semi-minor axis allowed within defined elliptical disk
    x_coord: 'number, ndarray'
    	x-coordinate in ellipse
    y_coord: 'number, ndarray'
    	y-coordinate in ellipse 
    """
    a = bd *el_constant1 + el_constant2 * bd**2 / bd_max
    result = x_coord**2 / a**2 + y_coord**2 / bd**2 - 1.
    #result = bd**2 / a**2 * (a**2 - x_coord**2) - y_coord**2
    #print(result)
    return result

def bd_equation(bd, x_coord, y_coord):
    """
    Equation of ellipse in the case that the provided point lies on the minor axis
    Function serves to be used in scipy.optimize.brenth to solve for bd 

    Parameters
    ----------
    bd: 'number'
    	semi-minor axis of ellipse
    x_coord: 'number, ndarray'
    	x-coordinate in ellipse
    y_coord: 'number, ndarray'
    	y-coordinate in ellipse 
    """
    a = x_coord
    result = x_coord**2 / a**2 + y_coord**2 / bd**2 - 1.
    #result = bd**2 / a**2 * (a**2 - x_coord**2) - y_coord**2
    #print(result)
    return result

def bd_solver(ell, xyz, z_sigma_lim, Hz, bd_max, el_constant1, el_constant2):
    """
    Function to solve for the ellipse equation to fit into form of ellipse_equation
    Chooses only to solve the equation numerically when necessary, avoiding the special cases.
    Funciton written in form to use with multiprocessing.pool and functools.partial

    Parameters
    ----------
    ell: 'int'
    	element number to iterate over
    xyz: 'ndarray with shape (3,N)'
    	xyz-coordinates
    z_sigma_lim: 'number'
    	sigma cuttoff to stop solving Ellipse equation for for z above a specified scale height threshold
    Hz: 'number'
    	Scale height along z axis
    bd_max: 'number'
    	Maximum semi-minor axis allowed within defined elliptical disk
    el_constant1: 'number'
    	First parameter for defining ellipse 
    el_constant2: 'number'
    	second parameter for defining ellipse
    """
    x_coord = xyz[0,ell]
    y_coord = xyz[1,ell]
    z_coord = xyz[2,ell]
    if z_coord > z_sigma_lim*Hz:
        res = bd_max+1.
    elif np.abs(y_coord) > bd_max:
        res = bd_max+1.
    elif np.abs(x_coord) > bd_max * (el_constant1 + el_constant2):
        res = bd_max+1.
    elif x_coord == 0.:
        if y_coord == 0.:
            res = 0
        else:
            res = y_coord
    elif y_coord == 0.:
        res = scipy.optimize.brenth(bd_equation, 0.000001, 1., 
                    args = (x_coord, y_coord))
    else:
        res = scipy.optimize.brenth(ellipse_equation, 0.000001, 1., 
                    args = (el_constant1, el_constant2,bd_max, x_coord, y_coord))
        if res<0:
            print(el_constant1,el_constant2,bd_max, x_coord, y_coord )
    return res


def EllipticalLBD(resolution, bd_max, Hz, z_sigma_lim, dens0,
                   velocity_factor, vel_0, el_constant1, el_constant2,
                   alpha, beta, theta, L_range, B_range, D_range, 
                   LSR_options={}, galcen_options = {}, return_all = False, **kwargs):
    """
    Creates kinematic disk following Elliptical Orbits of the from from Burton & Liszt (1982)
    Numerically solves for ellipse equation for every point within the disk space
    output is used directly to create a Longitude-Latitude-Velocity SpectralCube object using 'modspectra.cube.EllipticalLBD'

    Uses numexpr package to evaluate math
    Uses multiprocessing to solve Ellipse Equation

    Parameters
    ----------
    resolution: 'tuple, list'
    	Resolution to create grid 
    bd_max: 'number'
    	Maximum semi-minor axis allowed within defined elliptical disk
    Hz: 'number'
    	Scale height along z axis
    z_sigma_lim: 'number'
    	sigma cuttoff to stop solving Ellipse equation for for z above a specified scale height threshold
    dens0: 'number'
    	Density at midplane of Elliptical Disk
    velocity_factor: 'number'
    	Constant used to define velocity field in Burton & Liszt model
    vel_0: 'number'	
    	Max velocity of Elliptical orbit
    	Corresponds to velocity of outermost orbit on semi-minor axis
    el_constant1: 'number'
    	First parameter for defining ellipse 
    el_constant2: 'number'
    	second parameter for defining ellipse
    alpha: 'number'
    	Tilt angle alpha - see :class:'TiltedDisk'
    beta: 'number'
    	Tilt angle Beta - see :class:'TiltedDisk'
    	90 - beta is the inclination
    theta: 'number'
    	Tilt angle of major axis of Ellipse - see :class:'TiltedDisk'
    L_range: :list:'number'
    	Range of Longtiude to create grid over
    B_range: :list:'number'
    	Range of Latitude to create grid over
    D_range: :list:'number'
    	Range of Distances to create grid over

    LSR_options: 'dictionary', optional, must be keyword
    	Dictionary of **kwargs to pass into :class:'~astropy.coordinates.GalacticLSR'
    return_all: 'bool', optional, must be keyword
    	if True, will return all output components
    	used for diagnosing issues

    Returns
    -------
    lbd_coords_withvel: :class:'~astropy.coordinates.GalacticLSR'
    	astropy.coord array containing all coordinates corresponding to fabricated grid of points
    dens_grid: 'numpy.ndarray'
    	ndarray with shape (resolution) containing density of points in Longitude-Latitude-Distance grid
    	axes order swapped to be ready for SpectralCube creation
    cdelt: 'numpy.ndarray'
    	ndarray with shape (3) containing the step size for Longitude, Latitude, and Distance used in the grid
    	Used for WCS object creation in later instances

    disk_coordinates: :class:'TiltedDisk', optional, only if return_all == True
    	TiltedDisk coordinate class containing grid coordinates in original tilted disk space
    galcen_coords_withvel: :class:'~astropy.coordinates.Galactocentric'
    	TiltedDisk class transformed to Galactocentric frame 
    bd_grid: 'numpy.ndarray'
    	ndarray with shape (resolution) containing solved values of bd from Ellipse Equation solutions
    	axes order swapped to match dens_grid
    vel_magnitude_grid: 'numpy.ndarray'
    	ndarray with shape (resolution) contianing velocity vector magnitude at corresponding grid position
    """
    # Extract resolution information
    nx, ny, nz = resolution

    # Populate a uniform grid in Longitude-Latitude-Distance space
    lbd_grid = np.mgrid[L_range[0]:L_range[1]:nx*1j,
                        B_range[0]:B_range[1]:ny*1j,
                        D_range[0]:D_range[1]:nz*1j]
    # Transform grid into a 3 x N array for Longitude, Latitude, Distance axes                    
    lbd = lbd_grid.T.reshape(-1,3, order = "F").transpose()
    # Initiate astropy coordinates.Galactic object
    lbd_coords = coord.Galactic(l = lbd[0,:]*u.deg, b = lbd[1,:]*u.deg, distance = lbd[2,:]*u.kpc)

    if return_all:

    	# Convert regularized grid points into Galactocentric frame
    	galcen_coords = lbd_coords.transform_to(coord.Galactocentric(**kwargs))
    	disk_coords = galcen_coords.transform_to(TiltedDisk(alpha = alpha*u.deg, 
                                                        beta = beta*u.deg, theta = theta*u.deg))
    else:
    	disk_coords = lbd_coords.transform_to(TiltedDisk(alpha = alpha*u.deg, 
                                                        beta = beta*u.deg, theta = theta*u.deg))

    # Create standard numpy ndarray of disk_coords and reshape to grid, matching original lbd_grid object	
    disk_coords_arr = np.array([disk_coords.x.value, disk_coords.y.value, disk_coords.z.value])
    xyz_grid = disk_coords_arr.T.transpose().reshape(-1,nx,ny,nz)

    # initiate partial object to solve for Ellipse Equation
    partial_bd_solver = partial(bd_solver, xyz=disk_coords_arr, z_sigma_lim = z_sigma_lim, Hz = Hz, 
                        bd_max = bd_max, el_constant1 = el_constant1, el_constant2 = el_constant2)
    # Solve Ellipse Equation across multiple threads
    pool = multiprocessing.Pool()
    bd_vals = pool.map(partial_bd_solver, range(len(disk_coords.x.value)))
    pool.close()
    # Create grid of bd values solved from Ellipse Equation, matching original lbd_grid object
    bd_grid = np.array(bd_vals).T.transpose().reshape(nx,ny,nz)

    # Create grid of ad values (semi-major axis) derived from bd values
    ad_grid = ne.evaluate("bd_grid * (el_constant1 + el_constant2 * bd_grid / bd_max)")

    # Create grid of density values for the Elliptical Disk, mathcing original lbd_grid object
    dens_grid = np.zeros_like(bd_grid)
    z_coor = xyz_grid[2,:,:,:]
    dens_grid[(np.abs(z_coor)<(z_sigma_lim * Hz)) & (bd_grid<bd_max)] = dens0 * \
            np.exp(-0.5 * (z_coor[(np.abs(z_coor)<(z_sigma_lim * Hz)) & (bd_grid<bd_max)] / Hz)**2)

    # Solve for velocity magnitude using Angular Momentum and Velcotiy field from Burton & Liszt
    r_x = xyz_grid[0,:,:,:]
    r_y = xyz_grid[1,:,:,:]

    normalizer = ne.evaluate("1 / sqrt((r_x / ad_grid**2)**2 + (r_y / bd_grid**2)**2)")

    xtangent = ne.evaluate("r_y / bd_grid**2 * normalizer")
    ytangent = ne.evaluate("-r_x / ad_grid**2 * normalizer")
    # Angular momentum along the disk minor axis
    Lz_minor_axis = ne.evaluate("0. - bd_grid * vel_0 * (1. - exp(-bd_grid / velocity_factor))")  #r x v
    vel_magnitude_grid = ne.evaluate("abs(Lz_minor_axis / (r_x * ytangent - r_y * xtangent))")

    # Greate grid containing velocity vectors for orbits
    vel_xyz = np.zeros_like(xyz_grid)
    vel_xyz[0,:,:,:] = ne.evaluate("xtangent * vel_magnitude_grid")
    vel_xyz[1,:,:,:] = ne.evaluate("ytangent * vel_magnitude_grid")
    np.nan_to_num(vel_xyz)

    velocity_xyz = vel_xyz.T.reshape(-1,3, order = "F").transpose() * u.km/ u.s

    vel_cartesian = CartesianRepresentation(velocity_xyz)

    # Create new TiltedDisk object containing all newly calculated velocities
    disk_coordinates = TiltedDisk(x = disk_coords.x, y = disk_coords.y, z = disk_coords.z,
                v_x = vel_cartesian.x, v_y = vel_cartesian.y, v_z = vel_cartesian.z, 
                alpha = alpha*u.deg, beta = beta*u.deg, theta = theta*u.deg)

    # Transform to GalacticLSR frame
    if return_all:
    	galcen_coords_withvel = disk_coordinates.transform_to(coord.Galactocentric(**galcen_options))
    	lbd_coords_withvel = galcen_coords_withvel.transform_to(coord.GalacticLSR(**LSR_options))
    else:
    	lbd_coords_withvel = disk_coordinates.transform_to(coord.GalacticLSR(**LSR_options))

    # save Grid creation information for use in creating accurate WCS object associated with SpectralCube Object in future
    dD = lbd_grid[2,0,0,1] - lbd_grid[2,0,0,0]
    dB = lbd_grid[1,0,1,1] - lbd_grid[1,0,0,0]
    dL = lbd_grid[0,1,0,0] - lbd_grid[0,0,0,0]
    cdelt = np.array([dL, dB, dD])

    if return_all:
        return lbd_coords_withvel, np.swapaxes(dens_grid,0,2), cdelt, disk_coordinates, \
            galcen_coords_withvel, np.swapaxes(bd_grid,0,2), np.swapaxes(vel_magnitude_grid,0,2)
    else:
        return lbd_coords_withvel, np.swapaxes(dens_grid,0,2), cdelt



def EllipticalLBV(lbd_coords_withvel, density_gridin, cdelt, vel_disp, vmin, vmax, 
                    vel_resolution, L_range, B_range, species = 'hi', T_gas = 120. *u.K, memmap = False, da_chunks_xyz = 50):
    """
    Creates a Longitude-Latitude-Velocity SpectralCube object of neutral (HI 21cm) or ionized (H-Alpha) gas emission
    Uses output calculated from 'modspectra.cube.EllipticalLBD'

    Uses numexpr package to evaluate math

    Parameters
    ----------
    lbd_coords_withvel: :class:'~astropy.coordinates.GalacticLSR'
    	astropy.coord array containing all coordinates corresponding to fabricated grid of points
    dens_grid: 'numpy.ndarray'
    	ndarray with shape (resolution) containing density of points in Longitude-Latitude-Distance grid
    	axes order swapped to be ready for SpectralCube creation
    cdelt: 'numpy.ndarray'
    	ndarray with shape (3) containing the step size for Longitude, Latitude, and Distance used in the grid
    	Used for WCS object creation in later instances
    vel_disp: 'number, Quantity
    	Velocity dispersion of the gas in units of km/s (if not Quantity)
    vmin: 'number, Quantity'
    	Min Velocity to create in grid in units of km/s (if not Quantity)
    vmax: 'number, Quantity'
    	Max Velocity to create in grid in units of km/s (if not Quantity)
    vel_resolution: 'int'
    	Resolution to Create along velocity axis
    L_range: :list:'number'
    	Range of Longtiude to create grid over	
    B_range: :list:'number'
    	Range of Latitude to create grid over

    species: 'str', optional, must be keyword of either 'hi' or 'ha'
    	Specifies whether emission cube will be neutral (HI 21-cm) gas or ionized (H-Alpha) gas emission 
    	Defaults to HI netural gas
    T_gas: 'number, Quantity', optional, must be keyword
    	Temperature of neutral HI 21-cm emitting gas in Kelvin
    	Defaults to 120 K

    Returns
    -------
    emission_cube: :class:'numpy.ndarray'
        Emission values in LBD cube
    wcs: '~astropy.wcs.WCS'
        WCS information associated with emission_cube 
    """

    # Check for units
    if not isinstance(vel_disp, u.Quantity):
    	vel_disp = u.Quantity(vel_disp, unit = u.km / u.s)
    	logging.warning("No units specified for Velocity Dispersion, vel_disp, assuming"
    			"{}".format(vel_disp.unit))
    elif not vel_disp.unit == u.km/u.s:
    	vel_disp = vel_disp.to(u.km / u.s)

    if not isinstance(vmin, u.Quantity):
    	vmin = u.Quantity(vmin, unit = u.km / u.s)
    	logging.warning("No units specified for Min Velocity, vmin, assuming"
    			"{}".format(vmin.unit))
    elif not vmin.unit == u.km/u.s:
    	vmin = vmin.to(u.km / u.s)

    if not isinstance(vmax, u.Quantity):
    	vmax = u.Quantity(vmax, unit = u.km / u.s)
    	logging.warning("No units specified for Max Velocity, vmax, assuming"
    			"{}".format(vmax.unit))
    elif not vmax.unit == u.km/u.s:
    	vmax = vmax.to(u.km / u.s)

    if not isinstance(T_gas, u.Quantity):
    	T_gas = u.Quantity(T_gas, unit = u.K)
    	logging.warning("No units specified for T_gas, assuming"
    			"{}".format(T_gas.unit))
    elif not T_gas.unit == u.K:
    	T_gas = T_gas.to(u.K)



    # Define the lookup table of values of Sigma
    # First create my "lookup table" for the Gaussian evaluated at an array of sigma (5 sigma)

    # Extract resolution information
    nz, ny, nx = density_gridin.shape

    # Define the velocity channels
    VR, dv = np.linspace(vmin,vmax,vel_resolution, retstep=True)
    vr_grid = np.swapaxes(lbd_coords_withvel.radial_velocity.value.T.transpose().reshape(nx,ny,nz),0,2)

    # Calculate my sigma values
    vr_grid_plus = vr_grid[:,:,:,None]
    if memmap:
        gaussian_cells = da.exp(-1/2. * ((da.from_array(vr_grid_plus, chunks = (da_chunks_xyz,da_chunks_xyz,da_chunks_xyz,1)) - 
                                da.from_array(VR, chunks = -1)) / vel_disp.value)**2)
    else:
        gaussian_cells = ne.evaluate("exp(-1/2. * ((vr_grid_plus - VR) / vel_disp)**2)")

    # Calculate emission in each grid cell in Longitude-Latitude-Velocity space
    # Sums over Distance space
    dist = cdelt[2]
    if memmap:
        if species == 'hi':
            density_grid = da.from_array(density_gridin, chunks = (da_chunks_xyz,da_chunks_xyz,da_chunks_xyz)) * \
                                        33.52 / (T_gas.value * vel_disp.value) * dist * 1000. / 50.
            optical_depth = da.einsum('jkli, jkl->ijkl', gaussian_cells, density_grid).sum(axis = 1)
            result_cube = T_gas * (1  - da.exp(-1.* optical_depth))
            with ProgressBar():
                emission_cube = result_cube.compute() * u.K
        if species == 'ha':
            EM = da.from_array(density_gridin, chunks = (da_chunks_xyz,da_chunks_xyz,da_chunks_xyz)) **2 * dist * 1000.
            result_cube = da.einsum('jkli, jkl->ijkl', gaussian_cells, EM).sum(axis = 1)
            with ProgressBar():
                emission_cube = result_cube.compute() * u.cm**(-6) * u.pc
    else:
        if species == 'hi':
            density_grid = ne.evaluate("density_gridin *33.52 / (T_gas * vel_disp)* dist *1000. / 50.")
            optical_depth = np.einsum('jkli, jkl->ijkl', gaussian_cells, density_grid).sum(axis = 1)
            emission_cube = ne.evaluate("T_gas * (1 - exp(-1.* optical_depth))") * u.K
        if species =='ha':
            EM = ne.evaluate("density_gridin**2 * dist * 1000.")
            emission_cube = np.einsum('jkli, jkl->ijkl', gaussian_cells, EM).sum(axis = 1)
        
    # Create WCS Axes
    DBL_wcs = wcs.WCS(naxis = 3)
    DBL_wcs.wcs.crpix=[int(nx/2),int(ny/2),int(vel_resolution/2)]
    DBL_wcs.wcs.crval=[np.sum(L_range)/2, np.sum(B_range)/2, (vmax.value+vmin.value)/2]
    DBL_wcs.wcs.ctype=["GLON-CAR", "GLAT-CAR", "VRAD"]
    DBL_wcs.wcs.cunit=["deg", "deg", "km/s"]
    DBL_wcs.wcs.cdelt=np.array([cdelt[0], cdelt[1], dv.value])

    # Return Emission cube and WCS info
    return emission_cube, DBL_wcs


class EmissionCube(EmissionCubeMixin, SpectralCube):
    """
    Synthetic Emission cube container

    Parameters
    ----------
    data: 'ndarray', optional, if set, skips cube creation
        Data cube values
    wcs: 'astropy.wcs.WCS', optional, if set, skips cube creation
        WCS info
    meta: 'dict', optional, if set, skips cube creation
        Metadata
    mask: 'ndarray', optional, must be keyword
        Masking info
    fill_value: 'number', optional, must be keyword
        Values to fill for missing data
    header: 'dict'
        FITS header
    allow_huge_operations: 'bool'
        from SpectralCube
    beam: ''
        from SpectralCube
    wcs_tolerance: 'number'
        from SpectralCube

    resolution: 'tuple or list'
        Defines LBD Resolution of grid, must be shape (3)
    vel_resolution: 'int'
        Velocity axis resolution
    L_range: 'list, tuple, Quantity'
        Range of longitude in degrees to create grid [low, high]
    B_range: 'list, tuple, Quantity'
        Range of latitude in degrees to create grid [low, high]
    D_range: 'list,tuple,Quantity'
        Range of Distances in kpc to create grid [near, far]
    alpha: 'number, Quantity'
        First Tilt Angle of Disk
    beta: 'number, Quantity'
        Second Tilt Angle of Disk
        90 degrees - beta is the inclination
    theta: 'number, Quantity'
        Third tilt Angle of Disk for allignment of major axis with line of sight
    bd_max: 'number, Quantity'
        Max size of Ellipse along minor axis
        Default in units of kpc
    Hz: 'number, Quantity'
        Vertical Scale height of Disk
        Default in units of kpc
    z_sigma_lim: 'number'
        sigma cuttoff to stop solving Ellipse equation for for z above a specified scale height threshold
    dens0: 'number, Quantity'
        Density at midplane of Elliptical Disk
        Default units of cm^-3
    velocity_factor: 'number'
        Constant used to define velocity field in Burton & Liszt model
    vel_0: 'number'	
        Max velocity of Elliptical orbit
        Corresponds to velocity of outermost orbit on semi-minor axis
        Default unit of km/s
    el_constant1: 'number'
        First parameter for defining ellipse 
    el_constant2: 'number'
        second parameter for defining ellipse
    vel_disp: 'number, Quantity'
        Velocity dispersion of the gas in units of km/s (if not Quantity)
    vmin: 'number, Quantity'
        Min Velocity to create in grid in units of km/s (if not Quantity)
    vmax: 'number, Quantity'
        Max Velocity to create in grid in units of km/s (if not Quantity)

    species: 'str', optional, must be keyword of either 'hi' or 'ha'
        Specifies whether emission cube will be neutral (HI 21-cm) gas or ionized (H-Alpha) gas emission 
        Defaults to HI netural gas
    T_gas: 'number, Quantity', optional, must be keyword
        Temperature of neutral HI 21-cm emitting gas in Kelvin
        Defaults to 120 K
    LSR_options: 'dictionary', optional, must be keyword
        Dictionary of **kwargs to pass into :class:'~astropy.coordinates.GalacticLSR'
    return_all: 'bool', optional, must be keyword
        if True, will return all output components
        used for diagnosing issues
    BL82: 'bool', optional, must be keyword
        if True, will create a default cube with the exact parameters of Burton & Liszt 1982
    defaults: 'bool'
        if True, will use detaulf resolution and other information to create object
    create: 'bool'
        if True, will create cube using provided parameters
        ***Important*** must be True to create a cube, otherwise will assume it is 
            loading/initiating cube from provided data

    Returns
    -------
    SpectralCube BaseClass with EmissionCube container Class

    """


    def __init__(self, data = None, wcs = None, meta = None, 
                 mask = None, fill_value=np.nan, header=None, 
                 allow_huge_operations=False, beam=None, wcs_tolerance=0.0,
                 resolution = None, vel_resolution = None, 
                 L_range = None, B_range = None, D_range = None, 
                 alpha = None, beta = None, theta = None, 
                 bd_max = None, Hz = None, z_sigma_lim = None, dens0 = None,
                 velocity_factor = None, vel_0 = None, el_constant1 = None, el_constant2 = None, 
                 vel_disp = None, vmin = None, vmax = None, 
                 species = None, T_gas = None, LSR_options = {}, galcen_options = {}, return_all = False, 
                 BL82 = False, defaults = False, create = False, memmap = False, da_chunks_xyz = None, **kwargs):

        if not meta:
            meta = {}
        
        # Define Burton & Liszt 1982 parameters if needed
        if BL82:
            bd_max = 0.6 * u.kpc
            Hz = 0.1 * u.kpc
            z_sigma_lim = 3
            dens0 = 0.33 * 1/u.cm/u.cm/u.cm
            vel_0 = 360.*u.km/u.s
            velocity_factor = 0.1
            vel_disp = 9 * u.km/u.s
            el_constant1 = 1.6
            el_constant2 = 1.5
            T_gas = 120. * u.K
            alpha = 13.5 * u.deg
            beta = 20. * u.deg
            theta = 48.5 * u.deg

        # Define default parameters if needed
        if defaults:
            if not resolution:
                resolution = (64,64,64)
            if not vel_resolution:
                vel_resolution = 545
            if not L_range:
                L_range = [-10,10]*u.deg
            if not B_range:
                B_range = [-8,8] * u.deg
            if not D_range:
                D_range = [7,13] * u.kpc
            if not species:
                species = 'hi'
            if not vmin:
                vmin = -325. * u.km/u.s
            if not vmax:
                vmax = 325. * u.km/u.s

        if create:
            # Check units
            if not isinstance(L_range, u.Quantity):
                L_range = u.Quantity(L_range, unit = u.deg)
                logging.warning("No units specified for Longitude Range, assuming "
                        "{}".format(L_range.unit))
            elif not L_range.unit == u.deg:
                L_range = L_range.to(u.deg)

            if not isinstance(B_range, u.Quantity):
                B_range = u.Quantity(B_range, unit = u.deg)
                logging.warning("No units specified for Latitude Range, assuming "
                        "{}".format(B_range.unit))
            elif not B_range.unit == u.deg:
                B_range = B_range.to(u.deg)

            if not isinstance(D_range, u.Quantity):
                D_range = u.Quantity(D_range, unit = u.kpc)
                logging.warning("No units specified for Distance Range, assuming "
                        "{}".format(D_range.unit))
            elif not D_range.unit == u.kpc:
                D_range = D_range.to(u.kpc)

            if not isinstance(alpha, u.Quantity):
                alpha = u.Quantity(alpha, unit = u.deg)
                logging.warning("No units specified for Alpha, assuming "
                        "{}".format(alpha.unit))
            elif not alpha.unit == u.deg:
                alpha = alpha.to(u.deg)

            if not isinstance(beta, u.Quantity):
                theta = u.Quantity(beta, unit = u.deg)
                logging.warning("No units specified for Beta, assuming "
                        "{}".format(beta.unit))
            elif not beta.unit == u.deg:
                beta = beta.to(u.deg)

            if not isinstance(theta, u.Quantity):
                theta = u.Quantity(theta, unit = u.deg)
                logging.warning("No units specified for Theta, assuming "
                        "{}".format(theta.unit))
            elif not theta.unit == u.deg:
                theta = theta.to(u.deg)

            if not isinstance(bd_max, u.Quantity):
                bd_max = u.Quantity(bd_max, unit = u.kpc)
                logging.warning("No units specified for Max Semi-minor axis, bd_max, assuming "
                        "{}".format(bd_max.unit))
            elif not bd_max.unit == u.kpc:
                bd_max = bd_max.to(u.kpc)

            if not isinstance(Hz, u.Quantity):
                Hz = u.Quantity(Hz, unit = u.kpc)
                logging.warning("No units specified for Vertical Scale Height, Hz, assuming "
                        "{}".format(Hz.unit))
            elif not Hz.unit == u.kpc:
                Hz = Hz.to(u.kpc)

            if not isinstance(dens0, u.Quantity):
                dens0 = u.Quantity(dens0, unit = 1 / u.cm / u.cm / u.cm)
                logging.warning("No units specified for Midplane Density, dens0, assuming "
                        "{}".format(dens0.unit))
            elif not dens0.unit == 1 / u.cm / u.cm / u.cm:
                dens0 = dens0.to(1 / u.cm / u.cm / u.cm)

            if not isinstance(vel_0, u.Quantity):
                vel_0 = u.Quantity(vel_0, unit = u.km / u.s)
                logging.warning("No units specified for Max Velocity, vel_0, assuming "
                        "{}".format(vel_0.unit))
            elif not vel_0.unit == u.km/u.s:
                vel_0 = vel_0.to(u.km / u.s)

            if not isinstance(vel_disp, u.Quantity):
                vel_disp = u.Quantity(vel_disp, unit = u.km / u.s)
                logging.warning("No units specified for Velocity Dispersion, vel_disp, assuming "
                        "{}".format(vel_disp.unit))
            elif not vel_disp.unit == u.km/u.s:
                vel_disp = vel_disp.to(u.km / u.s)

            if not isinstance(vmin, u.Quantity):
                vmin = u.Quantity(vmin, unit = u.km / u.s)
                logging.warning("No units specified for Min Velocity, vmin, assuming "
                        "{}".format(vmin.unit))
            elif not vmin.unit == u.km/u.s:
                vmin = vmin.to(u.km / u.s)

            if not isinstance(vmax, u.Quantity):
                vmax = u.Quantity(vmax, unit = u.km / u.s)
                logging.warning("No units specified for Max Velocity, vmax, assuming "
                        "{}".format(vmax.unit))
            elif not vmax.unit == u.km/u.s:
                vmax = vmax.to(u.km / u.s)

            if (not da_chunks_xyz) and (memmap):
                da_chunks_xyz = 50
                logging.warning("Using a default chunksize of 50 per axis for memory mapping via dask")

            # Assign attributes
            self.bd_max = bd_max
            self.Hz = Hz
            self.z_sigma_lim = z_sigma_lim
            self.dens0 = dens0
            self.vel_0 = vel_0
            self.velocity_factor = velocity_factor
            self.vel_disp = vel_disp
            self.el_constant1 = el_constant1
            self.el_constant2 = el_constant2
            self.T_gas = T_gas
            self.alpha = alpha
            self.beta = beta
            self.theta = theta
            self.resolution = resolution
            self.vel_resolution = vel_resolution
            self.L_range = L_range
            self.B_range = B_range
            self.D_range = D_range
            self.species = species
            self.vmin = vmin
            self.vmax = vmax

            # Get LBD Grid Created
            self.LBD_output = EllipticalLBD(resolution, bd_max.value, Hz.value, 
                                                z_sigma_lim, dens0.value,
                                                velocity_factor, vel_0.value, el_constant1, el_constant2, 
                                                alpha.value, beta.value, theta.value, 
                                                L_range.value, B_range.value, D_range.value, 
                                                LSR_options = LSR_options, galcen_options = galcen_options, 
                                                return_all = return_all, **kwargs)

            if return_all:
                self.LBD_output_keys = ['lbd_coords', 'disk_density', 'cdelt', 
                                            'disk_coordinate_frame', 'galcen_coords',
                                            'bd_grid', 'vel_mag_grid']
            else:
                self.LBD_output_keys = ['lbd_coords', 'disk_density', 'cdelt']

            # Create LBV Cube
            data, wcs = EllipticalLBV(self.LBD_output[0], self.LBD_output[1], 
                                self.LBD_output[2], vel_disp, vmin, vmax, 
                                vel_resolution, L_range.value, B_range.value, 
                                species = species, T_gas = T_gas, memmap = memmap, da_chunks_xyz = da_chunks_xyz)

            # Metadata with unit info
            meta = {}
            meta['BUNIT'] = '{}'.format(data.unit)
        
        # Initialize Spectral Cube Object
        super().__init__(data = data, wcs = wcs, mask=mask, meta=meta, fill_value=fill_value,
                         header=header, allow_huge_operations=allow_huge_operations, beam=beam,
                         wcs_tolerance=wcs_tolerance, **kwargs)
        




    @classmethod
    def read(cls, file):
        cube = super().read(file)
        data = cube.unmasked_data[:]
        wcs = cube.wcs
        header = cube.header
        
        meta = {}
        if 'BUNIT' in header:
            meta['BUNIT'] = header['BUNIT']
        
        return cls(data = data, wcs = wcs, meta = meta, header = header)

        




