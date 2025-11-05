import pandas as pd
import numpy as np
import os
from copy import copy

from gp.gp_model import fit_gp


def get_prediction(target_date):
    if isinstance(target_date, str):
        target_date = pd.to_datetime(target_date)
    gparams = set_global_parameters()
    bounds = gparams['bounds']
    params_gp = {"FishType": 'Torsk', "kernel": 'Matern12', "likelihood_variance": 1,
                "kernel_variance": 1, "time_step_len": 1, "lags": 10,
                "lengthscales": 1}
    
    for gear in ['Trål', 'Krokredskap', 'Snurrevad', 'Garn']:
        vessel_data_path = f'/home/knowit/Home_Foresee/data/data_cloud/vessel_data/VMS_DCA_joined/recent_vms_dca_Torsk_{gear}.parquet'
        save_folder = f'/home/knowit/Home_Foresee/forseeModel/data/interpolated_fish_data/recent/Torsk_{gear}'
        if not os.path.exists(save_folder):
            print("SAVE: ", save_folder)
            os.makedirs(save_folder)
        
        df = pd.read_parquet(vessel_data_path)
        df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])#.dt.date
        min_date = target_date -  pd.to_timedelta(f'{params_gp["lags"]}D')
        window_width = params_gp['lags']
        steps = (max_date - min_date).days - window_width
        for w in range(steps):
            win_start = min_date + pd.to_timedelta(f'{w}D')
            win_end = win_start + pd.to_timedelta(f'{window_width}D')
            temp_df = df[df['TimeStamp'].between(win_start, win_end, inclusive='both')]
            if temp_df.shape[0] == 0:
                temp_df = pd.DataFrame({'TimeStamp':[np.NaN], 'Lat':pd.Series([np.NaN], dtype='float'),
                                        'Lon':pd.Series([np.NaN], dtype='float'), 'RW':pd.Series([np.NaN], dtype='float')})
            temp_df['WindowEnd'] = copy(win_end)
            print('----------------------------NUM POINTS: ', len(temp_df))                         
            fit_gp(df=temp_df, params=params_gp, bounds=bounds, grid_spacing=5_000, save_folder=save_folder)
            # fit_svgp(df=temp_df, params=params_gp, bounds=bounds, grid_spacing=10_000, save_folder=save_folder)
        

def main():
    # tf.keras.backend.clear_session()
    # tf.compat.v1.reset_default_graph()
    np.random.seed(42)
    gparams = set_global_parameters()
    bounds, ir, seed = gparams['bounds'], gparams['resolution'], gparams['seed']
    # chl_data_path, cop_data_path = gparams['chloro_data_path'], gparams['copernicus_data_path']
    # ves_data_paths, interpolated_data_path = gparams['vessel_data_paths'], gparams['interpolated_data_path']
    from_date, to_date = gparams['from_date'], gparams['to_date']
    params_gp = {"FishType": 'Torsk', "kernel": 'Matern12', "likelihood_variance": 1,
                "kernel_variance": 1, "time_step_len": 1, "lags": 10,
                "lengthscales": 1}
    
    for gear in ['Trål', 'Krokredskap', 'Snurrevad']:
        vessel_data_paths = [f'/home/knowit/Home_Foresee/data/data_cloud/vessel_data/VMS_DCA_joined/recent_vms_dca_Torsk_{gear}.parquet']
        save_folder = f'/home/knowit/Home_Foresee/forseeModel/data/interpolated_fish_data/Torsk_{gear}'
        if not os.path.exists(save_folder):
            print("SAVE: ", save_folder)
            os.makedirs(save_folder)
        
        df = pd.read_parquet(vessel_data_paths[0])
        df['TimeStamp'] = pd.to_datetime(df['TimeStamp']).dt.date
        min_date = pd.to_datetime('2022-08-01')
        max_date = pd.to_datetime('2022-12-12')
        window_width = params_gp['lags']
        steps = (max_date - min_date).days - window_width
        win_start = min_date
        for w in range(steps):
            win_end = win_start + pd.to_timedelta(f'{window_width}D')
            temp_df = df[df['TimeStamp'].between(win_start, win_end, inclusive='both')]
            if temp_df.shape[0] == 0:
                temp_df = pd.DataFrame({'TimeStamp':[np.NaN], 'Lat':pd.Series([np.NaN], dtype='float'),
                                        'Lon':pd.Series([np.NaN], dtype='float'), 'RW':pd.Series([np.NaN], dtype='float')})
            temp_df['WindowEnd'] = copy(win_end)
            print('----------------------------NUM POINTS: ', len(temp_df))                         
            fit_gp(df=temp_df, params=params_gp, bounds=bounds, grid_spacing=5_000, save_folder=save_folder)
            # fit_svgp(df=temp_df, params=params_gp, bounds=bounds, grid_spacing=10_000, save_folder=save_folder)
            win_start += pd.to_timedelta('1D')