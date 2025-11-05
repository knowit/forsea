from typing import Optional, Hashable, Iterable
from pandas._typing import ArrayLike
from torch import Tensor

import numpy as np
import pandas as pd
import pyproj
import torch
import xarray as xr

class ForSeaDataset(torch.utils.data.Dataset):  # type: ignore
    def __init__(
        self,
        ocean_data_path: str,
        route_data_path: str,
        fish_types: Iterable[str],
        gear_types: Iterable[str],
        mode: str = "regression",
        log_target: bool = False,
        thresholds: Iterable[float] = None,
        sequence_len: Optional[int] = None,
        batch_size: Optional[int] = 1,
        val_size: float = 0.2,
        shuffle: bool = False,
        gpu: Optional[torch.device | str] = None,
    ) -> None:
        self.dtype = torch.float32
        self.fish_types = fish_types
        self.gear_types = gear_types
        self.mode = mode
        self.log_target = log_target
        self.thresholds = thresholds
        self.sequence_len = sequence_len
        self.batch_size = batch_size
        self.val_size = val_size
        self.shuffle = shuffle
        self.gpu = gpu

        self.ocean_data_path = ocean_data_path
        self.ocean_data = xr.open_dataset(ocean_data_path).load()
        self.land_mask = torch.from_numpy(np.where(np.isnan(self.ocean_data.isel(time=0).vo), 0, 1))
        self._set_ocean_data_shape()
        self._set_ocean_minmax_values()

        self.dates = self.ocean_data.time.values.astype("datetime64[D]")
        self.n = len(self.dates)
        self.val_n = int(val_size * self.n)
        self.train_n = self.n - self.val_n 
        self.date_index_map = pd.Series(np.arange(self.n), index=self.dates)
        shuffled_index = np.random.permutation(self.n-sequence_len)
        self.train_index = shuffled_index[:self.train_n]
        self.val_index = shuffled_index[self.train_n:]

        # Function to transform lat-lon values to the projection of the grid data
        self.projection = pyproj.Transformer.from_crs(
            "EPSG:4326", self.ocean_data.proj_str, always_xy=True
        )

        self.route_data_path = route_data_path
        route_input, roundweight, route_date_index = self._load_route_data(
            route_data_path
        )
        self.num_route_features = route_input.shape[1]
        self.target_size = roundweight.shape[1]
        self.route_input = route_input
        self.target = self._calculate_target(roundweight)
        self.route_date_index = route_date_index
        self.shuffle_map = np.random.permutation(self.train_n)

    def __len__(self) -> int:
        n = (
            len(self.dates) - self.sequence_len + 1
            if self.sequence_len is not None
            else len(self.dates)
        )
        return n

    def __getitem__(
        self, idx: int
    ) -> (
        tuple[Tensor, Iterable[Tensor], Iterable[Tensor]]
        | tuple[Tensor, Tensor, Tensor]
    ):
        ocean_batch = self._get_ocean_tensor(idx)
        route_batch, target_batch = self._get_route_batch(idx)
        return ocean_batch, route_batch, target_batch
    
    def train_data(self):
        if self.shuffle:
            self._shuffle()
        for i in self.train_index:
            yield self.__getitem__(i)
    
    def val_data(self):
        for i in self.val_index:
            yield self.__getitem__(i)

    def _shuffle(self) -> None:
        self.train_index = np.random.permutation(self.train_index)

    def _calculate_target(self, roundweight: Tensor) -> Tensor:
        if self.mode in ("regression", "r"):
            if self.log_target:
                return torch.log1p(roundweight)
            else:
                return roundweight
        elif self.mode in ("classification", "c"):
            target = torch.zeros_like(roundweight)
            for i, t in enumerate(self.thresholds):
                target[:,i] = roundweight[:,i] > t
            return target
        else:
            raise ValueError(f"Invalid mode: {self.mode}.")

    def _set_ocean_data_shape(self) -> None:
        self.ocean_input_channels = len(self.ocean_data.keys())
        self.ocean_grid_shape = self.ocean_data.y.size, self.ocean_data.x.size
        if self.sequence_len is not None:
            if self.batch_size:
                self.ocean_data_shape = (
                    self.sequence_len,
                    self.batch_size,
                    self.ocean_input_channels,
                    *self.ocean_grid_shape,
                )
            else:
                self.ocean_data_shape = (
                    self.sequence_len,
                    self.ocean_input_channels,
                    *self.ocean_grid_shape,
                )
        else:
            if self.batch_size:
                self.ocean_data_shape = (
                    self.batch_size,
                    self.ocean_input_channels,
                    *self.ocean_grid_shape,
                )
            else:
                self.ocean_data_shape = (
                    self.ocean_input_channels,
                    *self.ocean_grid_shape,
                )

    def _set_ocean_minmax_values(self) -> None:
        ocean_min = {}
        ocean_max = {}
        for k in self.ocean_data.keys():
            ocean_min[k] = self.ocean_data[k].min().values
            ocean_max[k] = self.ocean_data[k].max().values
        self.ocean_min = ocean_min
        self.ocean_max = ocean_max

    def _load_route_data(self, path: str) -> tuple[Tensor, Tensor, ArrayLike]:
        route_data = pd.read_parquet(path).groupby("route_id").last().reset_index()

        grid_x, grid_y = self.ocean_data.x.values, self.ocean_data.y.values
        xmin, xmax = grid_x.min(), grid_x.max()
        ymin, ymax = grid_y.min(), grid_y.max()
        tmin, tmax = self.dates.min(), self.dates.max()

        start_x, start_y = self.projection.transform(
            route_data["start_lon"], route_data["start_lat"]
        )
        stop_x, stop_y = self.projection.transform(
            route_data["stop_lon"], route_data["stop_lat"]
        )
        route_data["start_x"] = start_x
        route_data["start_y"] = start_y
        route_data["stop_x"] = stop_x
        route_data["stop_y"] = stop_y

        route_data = route_data[route_data["start_x"].between(xmin, xmax)]
        route_data = route_data[route_data["start_y"].between(ymin, ymax)]
        route_data = route_data[route_data["stop_x"].between(xmin, xmax)]
        route_data = route_data[route_data["stop_y"].between(ymin, ymax)]
        route_data = route_data[route_data["stop_ts"].between(tmin, tmax)]

        # route_data['start_x'] = (route_data['start_x'] - xmin) / (xmax - xmin)
        # route_data['start_y'] = (route_data['start_y'] - ymin) / (ymax - ymin)

        route_data["stop_x"] = (route_data["stop_x"] - xmin) / (xmax - xmin)
        route_data["stop_y"] = (route_data["stop_y"] - ymin) / (ymax - ymin)

        route_data["date"] = route_data["stop_ts"].dt.normalize()
        # route_data['time_of_day'] = (route_data['stop_ts'] - route_data['date']).dt.total_seconds() / 86400
        # route_data['day_of_year'] = route_data['stop_ts'].dt.day_of_year / 365

        # route_data['distance']  = self._minmax_scale(route_data['distance'])
        # route_data['duration']  = self._minmax_scale(route_data['duration'])
        if self.gear_types is not None:
            route_data = route_data[route_data['gear_group'].isin(self.gear_types)]
        route_date_index = route_data["date"].map(self.date_index_map).values
        # input_columns = ['time_of_day', 'day_of_year', 'start_x', 'start_y', 'stop_x', 'stop_y', 'duration']
        input_columns = ["stop_x", "stop_y"]
        route_input = torch.from_numpy(route_data[input_columns].values)
        route_filter = torch.sum(route_input, dim=1) > 0
        # roundweight = torch.from_numpy(route_data[['roundweight']].values)
        roundweight = torch.from_numpy(
            route_data[self.fish_types].values
        )
        return route_input[route_filter], roundweight[route_filter], route_date_index[route_filter]

    def _get_ocean_array(
        self, var_key: Hashable, t_start: int, t_stop: Optional[int] = None
    ) -> np.ndarray:
        t_slice = slice(t_start, t_stop) if t_stop is not None else t_start
        return self.ocean_data[var_key].isel(time=t_slice).values

    def _scale_ocean_data(
        self, var_key: Hashable, ocean_array: np.ndarray
    ) -> np.ndarray:
        return (ocean_array - self.ocean_min[var_key]) / (
            self.ocean_max[var_key] - self.ocean_min[var_key]
        )

    def _get_ocean_tensor(self, idx: int) -> Tensor:
        ocean_array = np.zeros(self.ocean_data_shape)
        if self.sequence_len is not None:
            t_start = idx
            t_stop = idx + self.sequence_len
            for i, k in enumerate(self.ocean_data.keys()):
                # TODO: Add option for different barch sizes. Currently assumes batch_size is allways one.
                ocean_array[:, 0, i] = self._scale_ocean_data(
                    k, self._get_ocean_array(k, t_start, t_stop)
                )
        else:
            for i, k in enumerate(self.ocean_data.keys()):
                # TODO: Add option for different barch sizes. Currently assumes batch_size is allways one.
                ocean_array[0, i] = self._scale_ocean_data(
                    k, self._get_ocean_array(k, idx)
                )
        ocean_tensor = torch.from_numpy(ocean_array)
        ocean_tensor = torch.nan_to_num(ocean_tensor)
        return ocean_tensor.float().cuda()

    def _get_route_batch(
        self, idx: int
    ) -> tuple[Iterable[Tensor], Iterable[Tensor]] | tuple[Tensor, Tensor]:
        if self.sequence_len is not None:
            date_index = self.route_date_index == idx + self.sequence_len
            route_batch = self.route_input[date_index].float().cuda()
            target_batch = self.target[date_index].float().cuda()
            return route_batch, target_batch
            # routes = []
            # target = []
            # for t in range(self.sequence_len):
            #     date_index = self.route_date_index == idx + t
            #     route_batch = self.route_input[date_index].float().cuda()
            #     target_batch = self.target[date_index].float().cuda()
            #     routes.append(route_batch)
            #     target.append(target_batch)
            # return routes, target
        else:
            date_index = self.route_date_index == idx
            route_batch = self.route_input[date_index].float().cuda()
            target_batch = self.target[date_index].float().cuda()
            return route_batch, target_batch

    @staticmethod
    def _minmax_scale(feature: pd.Series) -> pd.Series:
        vmin = feature.min()
        vmax = feature.max()
        return (feature - vmin) / (vmax - vmin)

if __name__ == "__main__":
    config = {
        'lr': 0.01,
        'epochs': 100,
        "batch_accumulation": 32,
        'fish_types': ["Torsk", "Sei", "Hyse"],
        'gear_types': ["Trål"],
        'mode': "regression",
        "sequence_len": 30,
        'log_target': False,
        "thresholds": [1000, 1000, 1000],
        "criterion": "cross_entropy_with_logits",
        "optimizer": "adam",
        'model_params': {
            "input_dim": 10,
            "hidden_dim": [64, 128, 3],
            "kernel_size": (3, 3),
            "num_layers": 3,
            "batch_first": False,
            "return_all_layers": False,
        },
        'ocean_data_path': '../data/copernicus/datasets/norway.nc',
        'route_data_path': '../data/VMS_DCA_joined/catch_routes.parquet'
    }
    dataset = ForSeaDataset(
        config["ocean_data_path"],
        config["route_data_path"],
        fish_types=config["fish_types"],
        gear_types=config["gear_types"],
        mode=config["mode"],
        sequence_len=config["sequence_len"],
        log_target=config["log_target"],
        batch_size=1,
        shuffle=True,
    )
    for a,b,c in dataset.train_data():
        print(a.shape, b.shape, c.shape)