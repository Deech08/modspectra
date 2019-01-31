import pytest
import numpy as np 

# Set up the random number generator.
np.random.seed(1234)

def test_find_nearest_idx():
    '''
    Ensure function works
    '''
    from ..cube import find_nearest_idx
    arr = np.random.random(50)
    val = np.random.random()
    idx = find_nearest_idx(arr,val)[0]
    idx2 = np.argmin(np.abs(arr - val))
    assert np.allclose(idx, idx2)

def test_find_nannearest_idx():
    '''
    Ensure function works
    '''
    from ..cube import find_nannearest_idx
    arr = np.random.random(50)
    val = np.random.random()
    arr[[3,15,7,1,36,25]] = np.nan
    idx = find_nannearest_idx(arr,val)[0]
    idx2 = np.nanargmin(np.abs(arr - val))
    assert np.allclose(idx, idx2)

