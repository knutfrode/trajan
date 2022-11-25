"""
Extending xarray Dataset with functionality specific to trajectory datasets.

Presently supporting Cf convention H.4.1
https://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/cf-conventions.html#_multidimensional_array_representation_of_trajectories.
"""

import xarray as xr
import trajan as ta
import logging
from xhistogram.xarray import histogram

from .plot import Plot

logger = logging.getLogger(__name__)

@xr.register_dataset_accessor("traj")
class TrajAccessor:
    _obj = None
    __plot__ = None

    def __init__(self, xarray_obj):
        self._obj = xarray_obj


    @property
    def plot(self):
        if self.__plot__ is None:
            logger.debug(f'Setting up new plot object.')
            self.__plot__ = Plot(self._obj)

        return self.__plot__

    def gridtime(self, times):
        """Interpolate dataset to regular time interval"""

        # Remove drifter names, as mean cannot be applied to strings
        drifter_names = self._obj['drifter_names']
        self._obj = self._obj.drop_vars('drifter_names')

        d = xr.concat([self._obj.isel(trajectory=t).groupby_bins(
                       'time', bins=times).mean() for t in
                       range(self._obj.dims['trajectory'])], dim='trajectory')

        d['drifter_names'] = drifter_names
        return d
