Position-Position-Position Cubes
================================

`modspectra.cube` first relies on information provided by a density cube
in position-position-position space in Galactic Coordinates. for the models
presented in Krishnarao, Benjamin, & Haffner (2019), this is done via 
`~modspectra.cube.EllipticalLBD`::

    >>> from modspectra.cube import EllipticalLBD

The `EllipticalLBD` function is specific towards the tilted disk models
and uses the :class:`~modspectra.cube.TiltedDisk` coordinate frame. This function
can be used to derive some model based information, such as the Total Gas Mass. 
The example below will derive the total ionized gas mass along an elliptical ring 
structure as described in Krishnarao, Benjamin, & Haffner (2019)::

    # Import other dependencies
    >>> import numpy as np
    >>> import astropy.units as u
    >>> from astropy.coordinates.representation import CartesianDifferential

We start with defining the basic model parameters::

    # Define Model Parameters
    Hz = 0.26 # Midplane Scale Height at r = 0
    ne = 0.39 # Midplane density at r = 0
    Fz = 2.05 # Flaring Factor
    bd_max = 0.488 # Max Semi-minor axis of ellipse in kpc
    min_bd = bd_max / 2.
    el_constant1 = 1.6 # Geometric Factor
    el_constant2 = 1.5 # Geometric Factor
    alpha = 13.5 * u.deg # Tilt Angle
    beta = 20. * u.deg # 90 - inclination
    theta = 48.5 * u.deg # Major Axis Angle
    vel_0 = 360.*u.km/u.s # Max Tangential Velocity of elliptical orbits
    velocity_factor = 0.1 # Velocity Field helper term
    z_sigma_lim = 5. # Max number of scale heights to compute up to

    # Set PPP Space to compute model over
    L_range = [-10,10]*u.deg
    B_range = [-8,8] * u.deg
    D_range = [5,12] * u.kpc

    # Set some Default Galaxy Parameters
    galcen_distance = 8.127 * u.kpc
    v_bary = CartesianDifferential([0,0,0]*u.km/u.s)
    galcen_v_sun = CartesianDifferential([0,250.,0]*u.km/u.s)

    # Set model grid resolution
    resolution = (200,200,200) # Decrease for faster computation / less accuracy

`~modspectra.cube.EllipticalLBD` returns three items, the coordinates associated
with every point in the model grid in the `coord.GalacticLSR` frame, the density
grid in Distance-Latitude-Longitude space, and the distance spacing of the grid.
For now we will only worry about the first two output components::

    lbd_c,density_grid,_ = EllipticalLBD(resolution, 
                                         bd_max, 
                                         Hz, 
                                         z_sigma_lim, 
                                         ne, 
                                         velocity_factor, 
                                         vel_0, 
                                         el_constant1, 
                                         el_constant2,
                                         alpha.value, 
                                         beta.value, 
                                         theta.value, 
                                         L_range, 
                                         B_range, 
                                         D_range, 
                                         species = 'ha',
                                         galcen_options = {"galcen_v_sun":galcen_v_sun,
                                                           "galcen_distance":galcen_distance},
                                         LSR_options = {"v_bary":v_bary}, min_bd = min_bd,
                                         flaring = Fz, 
                                         flaring_radial = True,
                                         memmap = True)

    Computing Disk r-coordinate
    [########################################] | 100% Completed |  2.0s
    Computing Coordinates with Disk Velocity information in GalacticLSR Frame:
    [########################################] | 100% Completed |  1min 25.2s
    Computing Disk Density Grid:
    [########################################] | 100% Completed |  1min 18.2s
    Computing Ellipse Parameters
    [########################################] | 100% Completed |  1min 17.8s

    type(density_grid)

    <class 'dask.array.core.Array'>

Since we used the `memmap = True` keyword, the returned density array is a dask
array. We can load this into memory::

    density_grid = density_grid.compute()

Behind the scenes, `~modspectra.cube.EllipticalLBD` uses `numpy.mgrid` to construct
the ppp cube grid. We can do the same and use the grid to determine the volume of 
each cell in the grid ::

    nx,ny,nz = resolution
    # Create LBD Grid
    lbd_grid = np.mgrid[L_range[0]:L_range[1]:nx*1j,
                                    B_range[0]:B_range[1]:ny*1j,
                                    D_range[0]:D_range[1]:nz*1j]

    # Calculate step sizes in grid
    dD = np.abs(lbd_grid[2,0,0,0] - lbd_grid[2,0,0,1]) * u.kpc
    db = np.abs(lbd_grid[1,0,0,0] - lbd_grid[1,0,1,0]) * u.deg
    dl = np.abs(lbd_grid[0,0,0,0] - lbd_grid[0,1,0,0]) * u.deg

    # Distances are large enough compared to angles to use small angle approximation
    dD_l_grid = lbd_grid[2] * dl.to(u.rad) / u.rad # Projected distance along longitude axis
    dD_b_grid = lbd_grid[2] * db.to(u.rad) / u.rad # Projected distance along latitude axis

    # Calculate Volume of each cell in grid
    volume = dD_l_grid * dD_b_grid * dD

    # Change units and swap ordering to match Distance-Latitude_Longitude Array
    volume = np.swapaxes(volume.to(u.cm**3),0,2)

The mass of each cell can then be computed by multiplying these arrays with units::
    
    from astropy.constants import m_p as proton_mass

    # Calculate Mass of each cell
    Mass = density_grid*(u.cm**(-3)) * volume * proton_mass

    # Calculate total mass
    total_mass = Mass.sum()
    print("Total Mass = {:.3}".format(total_mass.to(u.solMass)))
    Total Mass = 1.15e+07 solMass

















