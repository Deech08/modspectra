import logging

from astropy import units as u
import numpy as np 
import matplotlib.pyplot as plt

from matplotlib.patches import Ellipse 

from matplotlib.colors import LogNorm

from astropy.visualization.wcsaxes.frame import RectangularFrame

def find_nannearest_idx(array,value):
    idx = np.nanargmin(np.abs(array-value))
    return [idx]

class EmissionCubeMixin(object):
	

    def ad(self, bd):
        """
        Calculates corresponding ellipse semi-major axis value given a semi-minor axis b
        """

        if not isinstance(bd, u.Quantity):
            bd = u.Quantity(bd, unit = u.kpc)

            logging.warning("No units specified for semi-minor axis, assuming"
                "{}".format(bd.unit))

        return bd * self.el_constant1 + self.el_constant2 * bd * bd / self.bd_max

    def xy_ellipse_patch(self):
        """
        Returns a '~matplotlib.patches.Ellipse' artist object of the Elliptical Disk in the xy plane

        Warning: Currently does not account for any alpha or beta tilt values...
        """
        return Ellipse(xy = [0.,0.], width = self.bd_max.value * 2., height = self.ad(self.bd_max).value * 2., angle = -self.theta.value)

    def lv_plot(self, latitude, swap_axes = False, fig = None, frame_class = RectangularFrame, aspect = 'auto',
                orientation = 'vertical', vmin = 5., vmax = 500., norm = LogNorm(), cmap = 'YlGnBu_r',
                invert_xaxis = False, invert_yaxis = False, spectral_unit = u.km/u.s, over_contour = None, 
                levels = (0.1, 0.4, 1.8, 3.8, 7., 11., 16., 24.), cmap_contour = 'Reds', 
                contour_options = {}, **kwargs):
        """
        Plot a Longitude-Velocity slice of the data at the specified latitude

        Parameters
        ----------

        latitude: 'Quantity or int'
            Latitude to extract slice from
            if 'int' then index value to slice at

        swap_axes: 'bool', optional, must be keyword
            if True, swaps x and y axes 
            Default is Longitude on y-axis and Velocity on x-axis
        fig: '~matplotlib.pyplot.figure', optional, must be keyword
            if provided, axis will be added to this figure instance
        frame_class: 'astropy.visualization.wcsaxes.frame', optional, must be keyword
            if provided, specifies astropy frame to use for Plot, such as EllipticalFrame
        aspect: 'str or int', optional, must be keyword
            aspect keyword used for 'matplotlib.pyplot.imshow'
        orientation: 'str', optional, must be keyword
            orientation of colorbar, keyword passed to 'matplotlib.pyplot.colorbar'
        vmin: 'number', optional, must be keyword
            min intensity to plot, keyword passed to 'matplotlib.pyplot.imshow'
        vmax: 'number', optional, must be keyword
            max intensity to plot, keyword passed to 'matplotlib.pyplot.imshow'
        norm: 'matplotlib.colors.', optional, must be keyword
            norm to pass into 'matplotlib.plt.imshow' for color scaling
        cmap: 'str', optional, must be keyword
            cmap keyword to pass into 'matplotlib.plg.imshow'
        invert_xaxis: 'bool', optional, must be keyword
            if True, will invert xaxis 
        invert_yaxis: 'bool', optional, must be keyword
            if True, will invert yaxis
        spectral_unit: 'astropy.units', optional, must be keyword
        	if provided, convert spectral axis to these units
        over_contour: 'EmissionCube or 'SpectralCube', optional, must be keyword
        	if provided, will overplot contours of this cube on image
        levels: 'list', optional, must be keyword
    		contour levels to plot or number of contours to plot 
    		passed to 'matplotlib.plt.contour'
    	cmap: 'str', optional, must be keyword
    		cmap keyword to pass into 'matplotlib.plt.contour'
    	contour_options: 'dict', optional, must be keyword
    		additional keywords to pass into 'matplotlib.plt.contour'
        """    

        # Initiate figure instance if needed
        if not fig:
            fig = plt.figure(figsize = (18,12))

        # Find Latitude slice index
        if isinstance(latitude, u.Quantity):
            vel_unit, lat_axis_values, lon_unit = self.world[int(self.shape[1]/2), :, int(self.shape[2]/2)]
            lat_slice = find_nannearest_idx(lat_axis_values, latitude)[0]
            if over_contour:
            	vel_unit_over, lat_axis_values_over, lon_unit_over = over_contour.world[int(over_contour.shape[1]/2), :, int(over_contour.shape[2]/2)]
            	lat_slice_over = find_nannearest_idx(lat_axis_values_over, latitude)[0]
            	wcs_over = over_contour[:,lat_slice_over,:].wcs
        else:
            lat_slice = latitude

        if not swap_axes:
        	data = self.hdu.data[:,lat_slice,:].transpose()
        	slices = ('y', lat_slice, 'x')
        	if over_contour:
        		wcs_over.nx = over_contour[:,lat_slice_over,:].header['NAXIS%i' % (1)]
        		wcs_over.ny = over_contour[:,lat_slice_over,:].header['NAXIS%i' % (2)]
        		data_over = over_contour.hdu.data[:,lat_slice_over,:].transpose()
        else:
        	data = self.hdu.data[:,lat_slice,:]
        	slices = ('x', lat_slice, 'y')
        	if over_contour:
        		data_over = over_contour.hdu.data[:,lat_slice_over,:]

        # Create wcs axis object
        ax = fig.add_subplot(111, projection = self.wcs, slices = slices, frame_class = frame_class)

        # Plot image
        im = ax.imshow(data, cmap = cmap, norm = norm, 
                vmin = vmin, vmax = vmax, aspect = aspect, **kwargs)

        if over_contour:
        	ct_transform = ax.get_transform(wcs_over)
        	ct = ax.contour(data_over, cmap = cmap_contour, levels = levels, transform = ct_transform, **contour_options)

        if invert_xaxis:
            ax.invert_xaxis()
        if invert_yaxis:
            ax.invert_yaxis()

        ax.coords[2].set_format_unit(spectral_unit)


        # Add axis labels
        lon_label = 'Galactic Longitude ' + '({})'.format(lon_unit.unit)
        vel_label = 'LSR Velocity ' + '({})'.format(spectral_unit)
        if swap_axes:
            ax.set_ylabel(vel_label, fontsize = 16)
            ax.set_xlabel(lon_label, fontsize = 16)
        else:
            ax.set_ylabel(lon_label, fontsize = 16)
            ax.set_xlabel(vel_label, fontsize = 16)

        # Add colorbar
        cbar = fig.colorbar(im, orientation = orientation)
        cbar.set_label('{}'.format(self.unit), size = 16)



    def lv_contour(self, latitude, swap_axes = False, fig = None, frame_class = RectangularFrame, aspect = 'auto', 
    				cmap = 'Reds', levels = (0.1, 0.4, 1.8, 3.8, 7., 11., 16., 24.), 
    				invert_xaxis = False, invert_yaxis = False, spectral_unit = u.km/u.s, **kwargs):
        """
        Plot contours of a Longitude-Velocity slice of the data at the specified latitude

        Parameters
        ----------

        latitude: 'Quantity or int'
        	Latitude to extract slice from
        	if 'int' then index value to slice at

        swap_axes: 'bool', optional, must be keyword
        	if True, swaps x and y axs 
        	Default is Longitude on y-axis and Velocity on x-axis
        fig: '~matplotlib.pyplot.figure', optional, must be keyword
        	if provided, axis will be added to this figure instance
        frame_class: 'astropy.visualization.wcsaxes.frame', optional, must be keyword
        	if provided, specifies astropy frame to use for Plot, such as EllipticalFrame
        aspect: 'str', optional, must be keyword
        	aspect keyword to pass into 'matplotlib.plt.contour'
        orientation: 'str', optional, must be keyword
        	keyword to pass into 'matplotlib.plt.colorbar'
        vmin: 'number', optional, must be keyword
        	min value to show - keyword to pass into 'matplotlib.plt.imshow'
        vmax: 'number', optional, must be keyword
        	max value to show - keyword to pass into 'matplotlib.plt.imshow'
        	cmap keyword to pass into 'matplotlib.plt.contour'
        levels: 'list', optional, must be keyword
        	contour levels to plot or number of contours to plot 
        	passed to 'matplotlib.plt.contour'
        invert_xaxis: 'bool', optional, must be keyword
            if True, will invert xaxis 
        invert_yaxis: 'bool', optional, must be keyword
            if True, will invert yaxis
        spectral_unit: 'astropy.units', optional, must be keyword
        	if provided, convert spectral axis to these units

        """ 

        # Initiate figure instance if needed
        if not fig:
            fig = plt.figure(figsize = (18,12))

        # Find Latitude slice index
        if isinstance(latitude, u.Quantity):
            vel_unit, lat_axis_values, lon_unit = self.world[int(self.shape[1]/2), :, int(self.shape[2]/2)]
            lat_slice = find_nannearest_idx(lat_axis_values, latitude)[0]
        else:
            lat_slice = latitude   

        if not swap_axes:
        	data = self.hdu.data[:,lat_slice,:].transpose()
        	slices = ('y', lat_slice, 'x')
        else:
        	data = self.hdu.data[:,lat_slice,:]
        	slices = ('x', lat_slice, 'y')

        # Create wcs axis object
        ax = fig.add_subplot(111, projection = self.wcs, slices = slices, frame_class = frame_class, aspect = aspect)

        # Plot contours
        ax.contour(data, cmap = cmap, levels = levels, **kwargs)

        if invert_xaxis:
            ax.invert_xaxis()
        if invert_yaxis:
            ax.invert_yaxis()

        ax.coords[2].set_format_unit(spectral_unit)


        # Add axis labels
        lon_label = 'Galactic Longitude ' + '({})'.format(lon_unit.unit)
        vel_label = 'LSR Velocity ' + '({})'.format(spectral_unit)
        if swap_axes:
            ax.set_ylabel(vel_label, fontsize = 16)
            ax.set_xlabel(lon_label, fontsize = 16)
        else:
            ax.set_ylabel(lon_label, fontsize = 16)
            ax.set_xlabel(vel_label, fontsize = 16)





