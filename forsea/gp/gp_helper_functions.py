import pandas as pd
import numpy as np
import itertools

# geometrisk transformasjon
import pyproj
from pyproj import CRS

# from shapely.geometry import Point
from global_land_mask import globe

from netCDF4 import Dataset, num2date, date2num

import gpflow

import numpy.ma as ma

def get_kernel(kernel_str):
    # kernels = {'Matern12': gpflow.kernels.Matern12(**{'name':'Matern12'}),
    #             'Matern32': gpflow.kernels.Matern32(name='Matern32'), 'Matern52':gpflow.kernels.Matern52(name='Matern52'),
    #             'Cosine': gpflow.kernels.Cosine(name='Cosine'), 'Constant':gpflow.kernels.Constant(name='Constant')}
    kernels = {'Matern12': gpflow.kernels.Matern12(**{'name':'Matern12'})}
    kernel_str_ls = kernel_str.split('+')
    kernel_str_ls = [k.split('*') for k in kernel_str_ls]
    print(kernel_str_ls)
    for i, kernel_pluss in enumerate(kernel_str_ls):
        for j, kernel_mult in enumerate(kernel_pluss):
            if j == 0:
                kernel_temp = kernels[kernel_mult]
            else:
                kernel_temp *= kernels[kernel_mult]
        if i == 0:
            kernel = kernel_temp
        else:
            kernel += kernel_temp
    return kernel  
    
def get_shape(bounds, N=10_000):
    (ymin, ymax), (xmin, xmax) = bounds
    x_range = abs(xmax - xmin)
    y_range = abs(ymax - ymin)
    aspect = x_range / y_range
    ny = int(np.sqrt(N / aspect))
    nx = int(ny * aspect)
    return ny, nx

def get_projection_grid(epsg, bounds, grid_spacing, round_bounds='outer'):
    # lat_bounds, lon_bounds = bounds['lat'], bounds['lon']
    (lat_min, lat_max), (lon_min, lon_max) = bounds['lat'], bounds['lon']
    t = pyproj.Transformer.from_crs('4326', epsg, always_xy=True)
    it = pyproj.Transformer.from_crs(epsg, '4326', always_xy=True)
    # y_bounds, x_bounds = t.transform((lat_min, lat_min, lat_max, lat_max), (lon_min, lon_max, lon_max, lon_min))
    x_bounds, y_bounds = t.transform((lon_min, lon_max, lon_max, lon_min), (lat_min, lat_min, lat_max, lat_max))
    ymin, ymax = np.min(y_bounds), np.max(y_bounds)
    xmin, xmax = np.min(x_bounds), np.max(x_bounds)

    if round_bounds == 'inner':
        ymin_idx = np.ceil(ymin / grid_spacing)
        ymax_idx = np.floor(ymax / grid_spacing)
        xmin_idx = np.ceil(xmin / grid_spacing)
        xmax_idx = np.floor(xmax / grid_spacing)
    elif round_bounds == 'outer':
        ymin_idx = np.floor(ymin / grid_spacing)
        ymax_idx = np.ceil(ymax / grid_spacing)
        xmin_idx = np.floor(xmin / grid_spacing)
        xmax_idx = np.ceil(xmax / grid_spacing)
    else:
        raise ValueError("round_bound must be either 'inner' or 'outer'")

    ny = int(ymax_idx - ymin_idx)
    nx = int(xmax_idx - xmin_idx)

    ymin = ymin_idx * grid_spacing
    ymax = ymax_idx * grid_spacing
    xmin = xmin_idx * grid_spacing
    xmax = xmax_idx * grid_spacing

    # ny, nx = get_shape(projected_bounds, grid_spacing=grid_spacing)
    # y_points = np.linspace(ymin, ymax, ny)
    y_points = np.linspace(ymax, ymin, ny)
    x_points = np.linspace(xmin, xmax, nx)
    y_grid, x_grid = np.meshgrid(y_points, x_points, indexing='ij')
    # lat_grid, lon_grid = it.transform(y_grid, x_grid)
    lon_grid, lat_grid = it.transform(x_grid, y_grid)
    ocean_mask = globe.is_ocean(lat_grid, lon_grid)
    return y_grid, x_grid, ocean_mask

def get_flattened_grid(y_grid, x_grid, ocean_mask):
    y_grid_flat = y_grid.ravel()
    x_grid_flat = x_grid.ravel()
    ocean_mask_flat = ocean_mask.ravel()
    time = np.zeros_like(y_grid_flat)
    grid_flat = np.column_stack([time, y_grid_flat, x_grid_flat])
    grid_flat[~ocean_mask_flat] = np.nan
    return grid_flat

def get_hourly_grid(y_grid, x_grid, ocean_mask):
    y_grid_flat = y_grid.ravel()
    x_grid_flat = x_grid.ravel()
    ocean_mask_flat = ocean_mask.ravel()
    n = len(y_grid_flat)
    hourly_grid = np.zeros((23*n, 3))
    for i, t in np.linspace(1, 0, 23):
        time = np.full_like(y_grid_flat, t)
        grid_flat = np.column_stack([y_grid_flat.copy(), x_grid_flat.copy(), time])
        hourly_grid
    grid_flat[~ocean_mask_flat] = np.nan
    return grid_flat

def get_7day_grid(y_grid, x_grid, ocean_mask, from_epsg, to_epsg):
    y_grid_flat = y_grid.flatten()
    x_grid_flat = x_grid.flatten()
    ocean_mask_flat = ocean_mask.flatten()

    if from_epsg != to_epsg:
        t = pyproj.Transformer.from_crs(from_epsg, to_epsg, always_xy=True)
        x_grid_flat, y_grid_flat = t.transform(x_grid_flat, y_grid_flat)

    y_grid_flat[~ocean_mask_flat] = np.nan
    x_grid_flat[~ocean_mask_flat] = np.nan
    n = len(y_grid_flat)
    encoded_times = np.arange(0, -7, -1)
    # encoded_times = -np.linspace(0, 1, 24)
    daily_grid = np.zeros((len(encoded_times)*n, 3))
    for i, t in enumerate(encoded_times):
        # print(daily_grid[n*i:n*(i+1), 1].shape, n*i, n*(i+1), n)
        daily_grid[n*i:n*(i+1), 0] = t
        daily_grid[n*i:n*(i+1), 1] = y_grid_flat.copy()
        daily_grid[n*i:n*(i+1), 2] = x_grid_flat.copy()
    return daily_grid, encoded_times

def get_spacetime_grid(y_grid, x_grid, time, ocean_mask, from_epsg, to_epsg):
    y_grid_flat = y_grid.flatten()
    x_grid_flat = x_grid.flatten()
    ocean_mask_flat = ocean_mask.flatten()

    if from_epsg != to_epsg:
        t = pyproj.Transformer.from_crs(from_epsg, to_epsg, always_xy=True)
        x_grid_flat, y_grid_flat = t.transform(x_grid_flat, y_grid_flat)

    y_grid_flat[~ocean_mask_flat] = np.nan
    x_grid_flat[~ocean_mask_flat] = np.nan
    n_space = len(y_grid_flat)
    n_time = len(time)
    spacetime_grid = np.zeros((n_space*n_time, 3))
    for i, t in enumerate(time):
        spacetime_grid[n_space*i:n_space*(i+1), 0] = t
        spacetime_grid[n_space*i:n_space*(i+1), 1] = y_grid_flat.copy()
        spacetime_grid[n_space*i:n_space*(i+1), 2] = x_grid_flat.copy()
    return spacetime_grid

def to_netcdf(path, dates, y_points, x_points, mean_function, std_function=None,
              epsg='4326', data_units=None, 
              time_units='seconds since 1970-01-01 00:00:00.0', 
              format='NETCDF4'): 
   
    if not hasattr(dates, '__len__'):
        dates = [dates]
    ny, nx, ndates = len(y_points), len(x_points), len(dates)

    if isinstance(dates[0], pd.Timestamp):
        try:
            dates = dates.to_pydatetime()
        except AttributeError:
            dates = [d.to_pydatetime() for d in dates]

    with Dataset(path, 'w', format=format) as nc_out:
        if epsg == '4326':
            ydimname, ydimname = 'lat', 'lon'
        else:
            ydimname, xdimname = 'y', 'x'

        nc_out.createDimension('time', None)
        nc_out.createDimension(ydimname, ny)
        nc_out.createDimension(xdimname, nx)

        
        timevar = nc_out.createVariable('time', 'int64', ('time',))
        yvar = nc_out.createVariable(ydimname, 'float32', (ydimname,))
        xvar = nc_out.createVariable(xdimname, 'float32', (xdimname,))

        # if gear is not None:
        #     nc_out.createDimension('gear', len(gear))
        #     gearvar = nc_out.createVariable('gear', '', (xdimname,))

        nc_out.epsg = epsg
        timevar.units = time_units
        yvar.units = 'degrees north' if epsg == '4326' else f'EPSG:{epsg}-Y'
        xvar.units = 'degrees east' if epsg == '4326' else  f'EPSG:{epsg}-X'

        yvar[:] = y_points
        xvar[:] = x_points
        timevar[:] = date2num(dates, units=timevar.units)

        meanvar = nc_out.createVariable('mean','float32',('time', ydimname, xdimname))
        if data_units is not None:
            meanvar.units = data_units
        meanvar[:] = mean_function

        if std_function is not None:
            stdvar = nc_out.createVariable('std','float32',('time', ydimname, xdimname))
            if data_units is not None:
                stdvar.units = data_units
            stdvar[:] = std_function
