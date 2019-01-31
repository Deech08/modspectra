import pytest
import matplotlib.pyplot as plt
import astropy.units as u
from ..cube import EmissionCube

test_cube = EmissionCube.create_LB82(resolution = (32,32,32), vel_resolution = 64)
test_cube_over = EmissionCube.create_LB82(resolution = (64,64,64), vel_resolution = 32)

def _run_lv_plot(latitude, **kwargs):
    cube = test_cube
    return cube.lv_plot(latitude, **kwargs)


def _run_lv_contour(latitude, **kwargs):
    cube = test_cube
    return cube.lv_contour(latitude, **kwargs)


BASELINE_DIR = 'baseline'

default_lat = 0.5 * u.deg

@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_basic():
    return _run_lv_plot(default_lat)

@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_lat_neg():
    return _run_lv_plot(-2.5 * u.deg)

@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_swap_axes():
    return _run_lv_plot(default_lat, swap_axes = True)

@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_data_range():
    return _run_lv_plot(default_lat, vmin = 0.01, vmax = 100.)

@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_norm_none():
    return _run_lv_plot(default_lat, vmin = 0.01, vmax = 10., norm = None)

@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_spectral_unit():
    return _run_lv_plot(default_lat, spectral_unit = u.kpc/u.Gyr)

@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_basic_contour():
    return _run_lv_contour(default_lat)

@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_contour_swap_axes():
    return _run_lv_contour(default_lat, swap_axes = True)

@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_levels():
    return _run_lv_contour(default_lat, levels = 5)

@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_over_true():
    return _run_lv_plot(default_lat, over_contour = True)

@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_over_other_cube():
    return _run_lv_plot(default_lat, over_contour = test_cube_over)

@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_over_other_cube_swap():
    return _run_lv_plot(default_lat, over_contour = test_cube_over, swap_axes = True)

@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_over_other_cube_horizontal():
    return _run_lv_plot(default_lat, over_contour = test_cube_over, orientation = "horizontal")

@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_over_other_cube_latidx():
    return _run_lv_plot(10, over_contour = test_cube_over)






