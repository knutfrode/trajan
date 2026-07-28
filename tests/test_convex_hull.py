import pytest
import numpy as np
import scipy
import trajan as ta


def test_convex_hull(openoil):
    ch = openoil.isel(time=-1).traj.convex_hull()
    assert isinstance(ch, scipy.spatial._qhull.ConvexHull)

    cha = openoil.isel(time=-1).traj.get_area_convex_hull()
    np.testing.assert_allclose(cha.values, 7.963e+09, rtol=1e-5)
