import xarray as xr
import numpy as np
import pyresample
import warnings
import os

from pyresample.geometry import AreaDefinition
from pyresample.bilinear import XArrayBilinearResampler

from typing import Optional, Iterable


class OceanDataResampler:
    """A class to resample and reproject gridded data to a new grid."""

    def __init__(self, grid_def_path: str) -> None:
        """Load the target area definition from the provided path."""
        self.target_area_def = pyresample.area_config.load_area(grid_def_path)

    def regrid(self, source_path: str, target_path: Optional[str] = None) -> xr.Dataset:
        """Load the source dataset, interpolate it to the target area, and return the new dataset.

        If target_path is provided, the new dataset is also saved to this path.
        """

        source_ds = xr.open_dataset(source_path)

        # Only use the first depth layer if depth is a dimension
        # We are ignoring depth as a dimension in the data as it makes the dataset much 
        # larger and reprojecting the data much more difficult.
        # In the future we may want to look into keeping depth as a dimension.
        if "depth" in source_ds:
            source_ds = source_ds.isel(depth=0)
        source_area_def = self._get_area_from_xarray(source_ds)

        print("Resampling...")
        resampler = XArrayBilinearResampler(
            source_area_def, self.target_area_def, 200e3
        )
        resampled_ds = xr.Dataset()
        # Resample one variable at a time and add it to the new dataset
        for var in source_ds.keys():
            print(var)
            result = resampler.resample(
                source_ds[var]
                .rename({"latitude": "y", "longitude": "x"})
                .isel(y=slice(None, None, -1))
            )
            resampled_var = xr.DataArray(
                result, result.coords, result.dims, var, source_ds[var].attrs
            )
            self._set_target_area_attrs(resampled_var)
            resampled_ds[var] = resampled_var

        self._set_target_area_attrs(resampled_ds)
        self._set_dimension_attrs(resampled_ds)
        print("Done.")
        if target_path is not None:
            resampled_ds.to_netcdf(target_path)
        return resampled_ds

    def regrid_merge(self, source_root: str, source_files: Iterable[str]) -> xr.Dataset:
        """Resample a list of datasets and merge them together."""
        datasets = []
        # Ignoring warnings because pyproj spams the output with complaints about proj strings
        # Information about the projection may be lost when converting from proj strings,
        # but this does not apply for the simple proj strings we are using.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for source_file in source_files:
                source_path = os.path.join(source_root, source_file)
                resampled_dataset = self.regrid(source_path)
                datasets.append(resampled_dataset)
        merged_data = xr.merge(datasets)
        return merged_data

    def _set_target_area_attrs(self, xr_object: xr.Dataset | xr.DataArray) -> None:
        """Set attributes related to the target area to a Dataset or DataArray."""
        xr_object.attrs["area_id"] = self.target_area_def.area_id
        xr_object.attrs["area_description"] = self.target_area_def.description
        xr_object.attrs["proj_str"] = self.target_area_def.proj_str
        xr_object.attrs["area_extent"] = self.target_area_def.area_extent

    def _set_dimension_attrs(self, xr_object: xr.Dataset | xr.DataArray) -> None:
        """Set dimension attributes from the target area to a Dataset or DataArray."""
        xr_object.x.attrs["step"] = self.target_area_def.resolution[1]
        xr_object.x.attrs["units"] = self.target_area_def.proj_dict["units"]
        xr_object.x.attrs["valid_min"] = self.target_area_def.area_extent[0]
        xr_object.x.attrs["valid_max"] = self.target_area_def.area_extent[2]

        xr_object.y.attrs["step"] = self.target_area_def.resolution[0]
        xr_object.y.attrs["units"] = self.target_area_def.proj_dict["units"]
        xr_object.y.attrs["valid_min"] = self.target_area_def.area_extent[1]
        xr_object.y.attrs["valid_max"] = self.target_area_def.area_extent[3]

    @staticmethod
    def _get_area_from_xarray(xr_dataset: xr.Dataset) -> AreaDefinition:
        """Create an AreaDefinition from a Xarray Dataset.

        Assumes the dataset is definded in WGS84 (aka. EPSG:4326) coordinates.
        """
        area_def = AreaDefinition(
            area_id="",
            description="",
            proj_id="",
            projection=f"EPSG:4326",
            width=xr_dataset.sizes["longitude"],
            height=xr_dataset.sizes["latitude"],
            area_extent=(
                xr_dataset.longitude.valid_min,
                xr_dataset.latitude.valid_min,
                xr_dataset.longitude.valid_max,
                xr_dataset.latitude.valid_max,
            ),
        )
        return area_def


if __name__ == "__main__":
    # Hindcast/reanalysis ocean data
    my_root = "/home/knowit/Home_Foresee/forseeModel/data/copernicus/my/copernicus-processed-data/"
    my_source_files = [
        "CMEMS-GLOBAL_001_029-chl_no3_nppv_o2_po4_si-2011_2020.nc",
        "CMEMS-GLOBAL_001_030-uo_vo_so_thetao-2011_2020.nc",
    ]

    # Forecast/analysis (Near Real-Time) ocean data
    nrt_root = "/home/knowit/Home_Foresee/forseeModel/data/copernicus/nrt/copernicus-processed-data/"
    nrt_files = [
        "CMEMS-GLOBAL_001_024-so-2020_2023.nc",
        "CMEMS-GLOBAL_001_024-uo_vo-2020_2023.nc",
        "CMEMS-GLOBAL_001_024-thetao-2020_2023.nc",
        "CMEMS-GLOBAL_001_028-several_vars-2020_2023.nc",
    ]

    area_def_path = "/home/knowit/Home_Foresee/forseeModel/forsea/geo/norway_grid.yaml"
    target_root = "/home/knowit/Home_Foresee/forseeModel/data/copernicus/datasets/"
    target_name = "norway"

    # Resample all datasets to same grid
    resampler = OceanDataResampler(area_def_path)
    my_data = resampler.regrid_merge(my_root, my_source_files)
    nrt_data = resampler.regrid_merge(nrt_root, nrt_files)

    # Only include variables that are included in both datasets
    variables = my_data.keys() & nrt_data.keys()
    my_data = my_data[variables]
    nrt_data = nrt_data[variables]

    # Remove dates in hindcast data that are also included in forecast data
    my_data = my_data.sel(
        time=slice(None, nrt_data.time.min() - np.timedelta64(1, "D"))
    )

    # Datasets are saved and reloaded to avoid running out of memory when concatenating the datasets.
    my_data.to_netcdf(os.path.join(target_root, f"{target_name}_my.nc"))
    my_data.close()
    nrt_data.to_netcdf(os.path.join(target_root, f"{target_name}_nrt.nc"))
    nrt_data.close()
    my_data = xr.open_dataset(
        "/home/knowit/Home_Foresee/forseeModel/data/copernicus/datasets/norway_my.nc"
    )
    nrt_data = xr.open_dataset(
        "/home/knowit/Home_Foresee/forseeModel/data/copernicus/datasets/norway_nrt.nc"
    )

    ocean_data = xr.concat([my_data, nrt_data], dim="time")
    ocean_data.to_netcdf(
        os.path.join(target_root, f"{target_name}.nc"),
        format="NETCDF4",
        engine="netcdf4",
    )
