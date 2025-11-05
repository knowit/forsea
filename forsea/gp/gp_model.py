import pandas as pd
import numpy as np
import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from sklearn.preprocessing import MinMaxScaler, StandardScaler

import tensorflow as tf
# import tensorflow_probability as tfp
# from tensorflow_probability import distributions as tfd
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
gpus = tf.config.list_physical_devices('GPU')
if len(gpus) == 0:
    raise SystemError('No GPUs found.')


# if gpus:
#   # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
#   try:
#     tf.config.set_logical_device_configuration(
#         gpus[0],
#         [tf.config.LogicalDeviceConfiguration(memory_limit=1024)])
#     logical_gpus = tf.config.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#   except RuntimeError as e:
#     # Virtual devices must be set before GPUs have been initialized
#     print(e)


import gpflow
from gpflow.ci_utils import reduce_in_tests
from gpflow.utilities import print_summary
from gp.gp_helper_functions import *
from interpolation.initialize_general_parameters import set_global_parameters
# gpflow.config.set_default_float(np.float32)
# gpflow.config.set_default_jitter(1.)
# print('GPU device:', tf.config.list_physical_devices('GPU'))
# gpu = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(device=gpu[0], enable=True)
# tf.config.experimental.set_memory_growth(device=gpu[1], enable=True)

gparams = set_global_parameters()
bounds, ir, seed = gparams['bounds'], gparams['resolution'], gparams['seed']
chl_data_path, cop_data_path = gparams['chloro_data_path'], gparams['copernicus_data_path']
ves_data_paths, interpolated_data_path = gparams['vessel_data_paths'], gparams['interpolated_data_path']
from_date, to_date = gparams['from_date'], gparams['to_date']
epsg = '3035'

class SVGPModel:
    def __init__(self, data_path, fish_types, gear_types):
        self.data_path = data_path
        self.fish_types = fish_types
        self.gear_types = gear_types

    def fit_models(M, iterations=20_000):
        pass

def fit_gp(df, params, bounds, save_folder, iterations=20000, grid_spacing = 5000):


    df = df.dropna()
    df['RW'] = np.log(df['RW'] + 1e-2)
    # likelihood_variance = params['likelihood_variance']
    # kernel_variance = params['kernel_variance']
    # time_step_len = params['time_step_len']
    # np.random.seed(seed) 
    # kf.variance.assign(kernel_variance)
    # kf.lengthscales.assign(np.array([1e-5, 1e-5, 1e-5]))
    
    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'], format='%Y-%m-%d')
    df['WindowEnd'] = pd.to_datetime(df['WindowEnd'], format='%Y-%m-%d')

    df = df.sort_values(by='TimeStamp', ascending=False).reset_index(drop=True)
    out_time = df['WindowEnd'].max()
    days = np.array([(out_time - t).total_seconds()/86400 for t in list(df['TimeStamp'])])

    df['Days'] = days

    X = np.zeros((len(df), 3), dtype='float32')
    t = pyproj.Transformer.from_crs('4326', epsg, always_xy=True)
    # crs_y, crs_x = t.transform(df['Lat'], df['Lon'])
    crs_x, crs_y = t.transform(df['Lon'], df['Lat'])
    X[:,0] = days
    X[:,1] = crs_y
    X[:,2] = crs_x

    z = np.expand_dims(df['RW'].to_numpy(), 1)
    X, z = X.astype(np.float64), z.astype(np.float64)
    scaler_outp = StandardScaler()
    
    z = scaler_outp.fit_transform(z)


    # kf = get_kernel(params['kernel'])
    # lengthscales = [np.std(days), 4e5, 4e5]
    lengthscales = [np.std(days), np.std(crs_y), np.std(crs_x)]
    # kf = gpflow.kernels.Matern12(lengthscales=[0.5, 1e4, 1e4], variance=0.5)
    # kf = gpflow.kernels.Matern32(lengthscales=[0.5, 1e4, 1e4])

    print('>>>>>>>>>>>>>>>>>', out_time)
    print(np.std(days), np.std(crs_y), np.std(crs_x))
    print(np.std(days / lengthscales[0]), np.std(crs_y / lengthscales[1]), np.std(crs_x / lengthscales[2]))
    kf = gpflow.kernels.Matern12(lengthscales=lengthscales, variance=5)
    # model = gpflow.models.GPR((X, z), kernel=kf, mean_function=gpflow.mean_functions.Constant(0))
    model = gpflow.models.GPR((X, z), kernel=kf, mean_function=None)
    # model.likelihood.variance.assign(.33)
    # model.kernel.variance.assign(300.)
    # print_summary(model)
    # model.likelihood.variance.assign(likelihood_variance)
    training_loss = model.training_loss

    # gpflow.set_trainable(model.kernel.lengthscales, False)
    # gpflow.set_trainable(model.kernel.variance, True)
    # gpflow.set_trainable(model.likelihood.variance, True)
    gpflow.set_trainable(model.mean_function, False)

    optimizer = gpflow.optimizers.Scipy()
    optimizer.minimize(training_loss, model.trainable_variables, options=dict(maxiter=iterations), method='L-BFGS-B', tol=1e-6)  
    # optimizer.minimize(training_loss, model.trainable_variables, options=dict(maxiter=iterations))  
    print_summary(model)
    # for repeat in range(5):
    #     optimizer.minimize(training_loss, model.trainable_variables, options=dict(maxiter=iterations), method='L-BFGS-B', tol=1e-6)  

    y_grid, x_grid, ocean_mask = get_projection_grid(epsg, bounds, grid_spacing=grid_spacing)
    grid_flat = get_flattened_grid(y_grid, x_grid, ocean_mask)
    posterior = model.posterior()
    field_flat, var_flat = posterior.predict_f(grid_flat)
    field_flat = field_flat.numpy()
    field_flat = scaler_outp.inverse_transform(field_flat)
    # print_field(field_flat)
    field_flat = np.exp(field_flat) - 1e-2
    field_flat[(field_flat < 0) & (field_flat != np.NaN)] = 0
    
    field = field_flat.reshape(y_grid.shape)
    field = np.ma.masked_array(field, ~ocean_mask)
    field = np.expand_dims(field, 0)
    # field = tf.reshape(field_flat, y_grid.shape)
    save_path = f'{save_folder}/fld_RW_{out_time.strftime("%Y-%m-%d")}.nc'
    to_netcdf(save_path, field, y_grid[:,0], x_grid[0], [out_time], epsg=epsg)
    return field, y_grid, x_grid               

def get_7day_prediction(target_date, bounds, lookback, iterations=20000, grid_spacing = 5000, sparse=False, M=200):

    if isinstance(target_date, str):
        target_date = pd.to_datetime(target_date)
    fish_type = 'Torsk'
    for gear in ['Trål', 'Krokredskap', 'Snurrevad']:
        # vessel_data_path = f'/home/knowit/Home_Foresee/data/data_cloud/vessel_data/VMS_DCA_joined/recent_vms_dca_{fish_type}_{gear}.parquet'
        vessel_data_path = f'/home/knowit/Home_Foresee/data/data_cloud/vessel_data/VMS_DCA_joined/vms_dca_2022_{fish_type}_{gear}.parquet'
        save_folder = f'/home/knowit/Home_Foresee/forseeModel/data/interpolated_fish_data/recent/{fish_type}_{gear}'
        if not os.path.exists(save_folder):
            print("SAVE: ", save_folder)
            os.makedirs(save_folder)
        
        data = pd.read_parquet(vessel_data_path)
        data['time'] = pd.to_datetime(data['time'])
        window_start = (target_date -  pd.Timedelta(days=lookback))
        df = data[data['time'].between(window_start, target_date, inclusive='both')]
        # date_index = (data['time'] >= window_start) & (data['time'] <= target_date)
        # print(gear, np.sum(data['time'] >= window_start), np.sum(data['time'] <= target_date))
        # print(data['time'].min(), data['time'].max(), '---------', window_start, target_date, np.sum(date_index))
        # df = data[date_index]
        if df.shape[0] == 0:
            raise ValueError('Dataframe is empty')
        df = df.dropna()
        df = df[df['distance'] > 0]
        df = df[df['duration'] > 0]
        print('----------------------------NUM POINTS: ', len(df), window_start, target_date)                         
        # df['rw'] = np.log(df['rw'] / (df['distance'] * df['duration']) + 1e-2)
        df['rw'] = np.log(df['rw'] / df['distance'] + 1e-2)
        df = df.sort_values(by='time', ascending=False).reset_index(drop=True)
        days = ((target_date - df['time']) / pd.Timedelta(seconds=1)) / 86400
        X = np.zeros((len(df), 3), dtype='float64')
        t = pyproj.Transformer.from_crs('4326', epsg)
        crs_y, crs_x = t.transform(df['lat'], df['lon'])
        X[:,0] = days
        X[:,1] = crs_y
        X[:,2] = crs_x

        z = np.expand_dims(df['rw'].to_numpy(), 1)
        z = z.astype('float64')
        scaler_outp = StandardScaler()
        z = scaler_outp.fit_transform(z)
        lengthscales = [np.std(days), np.std(crs_y), np.std(crs_x)]
        print('>>>>>>>>>>>>>>>>>', target_date)
        print(np.std(days), np.std(crs_y), np.std(crs_x))
        print(np.std(days / lengthscales[0]), np.std(crs_y / lengthscales[1]), np.std(crs_x / lengthscales[2]))


        if sparse:
            N = len(z)
            Xm = X[-M:,:].copy()
            minibatch_size = 100

            kernel = gpflow.kernels.Matern12(lengthscales=lengthscales, variance=5)
            # kf = gpflow.kernels.Matern32(lengthscales=lengthscales, variance=5)
            model = gpflow.models.SVGP(kernel, gpflow.likelihoods.Gaussian(), Xm, num_data=N)
            # model = gpflow.models.SVGP((X, z), kernel, Xm)
            # model = gpflow.models.GPR((X, z), kernel=kernel, mean_function=None)
            gpflow.set_trainable(model.mean_function, False)
            print_summary(model)
            # training_loss = model.training_loss
            train_dataset = tf.data.Dataset.from_tensor_slices((X, z)).repeat().shuffle(N)
            train_iter = iter(train_dataset.batch(minibatch_size))
            # training_loss = model.training_loss_closure(train_iter, compile=True)
            training_loss = model.training_loss_closure((X, z), compile=True)
            # training_loss = model.training_loss((X, z))

            optimizer = gpflow.optimizers.Scipy()
            optimizer.minimize(training_loss, model.trainable_variables, options=dict(maxiter=iterations), method='L-BFGS-B', tol=1e-6)  
            print_summary(model)

        else:
            kernel = gpflow.kernels.Matern12(lengthscales=lengthscales, variance=5)
            # kf = gpflow.kernels.Matern32(lengthscales=lengthscales, variance=5)
            model = gpflow.models.GPR((X, z), kernel=kernel, mean_function=None)
            training_loss = model.training_loss
            gpflow.set_trainable(model.mean_function, False)
            optimizer = gpflow.optimizers.Scipy()
            optimizer.minimize(training_loss, model.trainable_variables, options=dict(maxiter=iterations), method='L-BFGS-B', tol=1e-6)
            
            f64 = gpflow.utilities.to_default_float
            model.kernel.lengthscales.prior = tfd.Gamma(f64(1.0), f64(1.0))
            model.kernel.variance.prior = tfd.Gamma(f64(1.0), f64(1.0))
            model.likelihood.variance.prior = tfd.Gamma(f64(1.0), f64(1.0))
            # model.mean_function.A.prior = tfd.Normal(f64(0.0), f64(10.0))
            # model.mean_function.b.prior = tfd.Normal(f64(0.0), f64(10.0))
            print_summary(model)

            num_burnin_steps = reduce_in_tests(300)
            num_samples = reduce_in_tests(1000)

            # Note that here we need model.trainable_parameters, not trainable_variables - only parameters can have priors!
            hmc_helper = gpflow.optimizers.SamplingHelper(
                model.log_posterior_density, model.trainable_parameters
            )

            hmc = tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=hmc_helper.target_log_prob_fn,
                num_leapfrog_steps=10,
                step_size=0.01,
            )
            adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
                hmc,
                num_adaptation_steps=10,
                target_accept_prob=f64(0.75),
                adaptation_rate=0.1,
            )

            @tf.function
            def run_chain_fn():
                return tfp.mcmc.sample_chain(
                    num_results=num_samples,
                    num_burnin_steps=num_burnin_steps,
                    current_state=hmc_helper.current_state,
                    kernel=adaptive_hmc,
                    trace_fn=lambda _, pkr: pkr.inner_results.is_accepted,
                )


            samples, traces = run_chain_fn()
            parameter_samples = hmc_helper.convert_to_constrained_values(samples)

            param_to_name = {
                param: name
                for name, param in gpflow.utilities.parameter_dict(model).items()
            }
            print_summary(model)


        y_grid, x_grid, ocean_mask = get_projection_grid(epsg, bounds, grid_spacing=grid_spacing)
        grid_flat = get_7day_grid(y_grid, x_grid, ocean_mask)
        posterior = model.posterior()
        field_flat, var_flat = posterior.predict_f(grid_flat)
        field_flat = field_flat.numpy()
        field_flat = scaler_outp.inverse_transform(field_flat)
        field_flat = np.exp(field_flat) - 1e-2
        field_flat[(field_flat < 0) & (field_flat != np.NaN)] = 0
        
        field = field_flat.reshape((7, *y_grid.shape))
        field = np.ma.masked_equal(field, np.nan)
        # field = tf.reshape(field_flat, y_grid.shape)
        save_path = f'{save_folder}/sparse_RW_{target_date.strftime("%Y-%m-%d")}.nc'
        prediction_dates = [target_date + pd.Timedelta(days=d) for d in range(7)]
        to_netcdf(save_path, field, prediction_dates, y_grid[:,0], x_grid[0], epsg=epsg)


def fit_daily_models(data_path, save_path, 
                     target_date, fish_types, gear_types, lookback, 
                     bounds, model_epsg='3035', grid_epsg=None, grid_spacing=5000, sparse=True,
                     M=200, iterations=20000, minibatch_size=100, 
                     time_column='time', normalize_dates=False, float_type='float64'):

    all_data = pd.read_parquet(data_path)
    all_data[time_column] = pd.to_datetime(all_data[time_column])
    
    if target_date == 'today':
        target_date = pd.to_datetime('today')
    elif target_date == 'latest':
        target_date = all_data[time_column].max().normalize()
    elif isinstance(target_date, str):
        target_date = pd.to_datetime(target_date)

    if normalize_dates:
        target_date = target_date.normalize()
        
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if grid_epsg is None:
        grid_epsg = model_epsg

    y_grid, x_grid, ocean_mask = get_projection_grid(grid_epsg, bounds, grid_spacing=grid_spacing)
    grid_flat, encoded_times = get_7day_grid(y_grid, x_grid, ocean_mask, from_epsg=grid_epsg, to_epsg=model_epsg)
    prediction_dates = target_date - pd.to_timedelta(encoded_times * 86400, unit='S')

    # prediction_dates = target_date + pd.to_timedelta(np.arange(7), unit='h')
    # prediction_dates_encoded = ((target_date - prediction_dates) / pd.Timedelta(seconds=1)) / 86400
    # print(prediction_dates_encoded)
    # grid_flat = get_spacetime_grid(y_grid, x_grid, prediction_dates_encoded, ocean_mask, from_epsg=grid_epsg, to_epsg=model_epsg)

    # prediction_dates = [target_date + pd.Timedelta(days=d) for d in range(7)]
    all_data['time_encoded'] = ((target_date - all_data[time_column]) / pd.Timedelta(seconds=1)) / 86400
    n_prediction_dates = len(prediction_dates)
    for fish in fish_types:
        for gear in gear_types:
            data = all_data[all_data['gear'] == gear].copy()
            try:
                # data = data[data[fish] > 0]
                data['rw'] = data[fish] / data['n']
            except KeyError as ke:
                print(ke)
                continue
            
            # try:
            current_save_path = os.path.join(save_path, f'RW_{fish}_{gear}.nc')
            if normalize_dates:
                data[time_column] = data[time_column].dt.normalize()
            window_start = (target_date - pd.Timedelta(days=lookback))
            df = data[data[time_column].between(window_start, target_date, inclusive='both')]
            if df.shape[0] == 0:
                # raise ValueError('Dataframe is empty')
                continue
            
            df = df.dropna()
            print(f'{fish}, {gear} - Num data points: {len(df)}')
            # df = df[df['distance'] > 0]
            # df = df[df['duration'] > 0]
            # df['rw'] = np.log(df['rw'] / (df['distance'] * df['duration']) + 1e-2)
            # df['rw'] = np.log(df['rw'] / df['distance'] + 1e-2)
            df['rw'] = np.log(df['rw'] + 1e-2)
            df = df.sort_values(by=time_column, ascending=False).reset_index(drop=True)
            
            X = np.zeros((len(df), 3), dtype=float_type)
            t = pyproj.Transformer.from_crs('4326', model_epsg, always_xy=True)
            # crs_y, crs_x = t.transform(df['lat'], df['lon'])
            crs_x, crs_y = t.transform(df['lon'], df['lat'])
            X[:,0] = df['time_encoded']
            X[:,1] = crs_y
            X[:,2] = crs_x
            z = np.expand_dims(df['rw'].to_numpy(), 1)
            scaler_outp = StandardScaler()
            z = scaler_outp.fit_transform(z)
            z = z.astype(float_type)
            # lengthscales = np.array([np.std(df['time_encoded']), np.std(crs_y), np.std(crs_x)], dtype=float_type)
            lengthscales = np.array([1, 24_000, 24_000], dtype=float_type)
            # lengthscales = [l.astype(np.float32) for l in lengthscales]
            N = len(z)
            Xm = X[-M:,:].copy()
            # print(X.dtype)
            # print(X)
            # print(z.dtype)
            # print(z)
            # print(lengthscales.dtype)
            # print(lengthscales)
            # kernel = gpflow.kernels.Matern12(lengthscales=lengthscales)
            kernel = gpflow.kernels.Matern52(lengthscales=lengthscales)
            if sparse:
                model = gpflow.models.SVGP(kernel, gpflow.likelihoods.Gaussian(np.float64(1e-2)), Xm, num_data=N)
                gpflow.set_trainable(model.mean_function, False)
                # gpflow.set_trainable(model.kernel.lengthscales, False)
                train_dataset = tf.data.Dataset.from_tensor_slices((X, z)).repeat().shuffle(N)
                train_iter = iter(train_dataset.batch(minibatch_size))
                training_loss = model.training_loss_closure(train_iter, compile=True)
                # training_loss = model.training_loss_closure((X, z), compile=True)
                # training_loss = model.training_loss((X, z))

                optimizer = gpflow.optimizers.Scipy()
                optimizer.minimize(training_loss, model.trainable_variables, options=dict(maxiter=iterations), method='L-BFGS-B', tol=1e-6)
            else:
                model = gpflow.models.GPR((X, z), kernel=kernel)
                gpflow.set_trainable(model.mean_function, False)
                # gpflow.set_trainable(model.kernel.lengthscales, False)
                training_loss = model.training_loss
                optimizer = gpflow.optimizers.Scipy()
                optimizer.minimize(training_loss, model.trainable_variables, options=dict(maxiter=iterations), method='L-BFGS-B', tol=1e-6)

            # with open(current_save_path, 'wb') as fp:
            #     pickle.dump(model.read_trainables(), fp)
            print_summary(model)
            print('---------------FIT SUCCESSFUL')
            posterior = model.posterior()
            mean_flat, var_flat = posterior.predict_f(grid_flat)

            mean_flat = mean_flat.numpy()
            mean_flat = scaler_outp.inverse_transform(mean_flat)
            mean_flat = np.exp(mean_flat) - 1e-2
            mean_flat[(mean_flat < 0) & (mean_flat != np.NaN)] = 0
            mean_function = mean_flat.reshape((n_prediction_dates, *y_grid.shape))
            mean_function = np.ma.masked_invalid(mean_function)
            
            var_flat = var_flat.numpy()
            std_flat = np.sqrt(var_flat)
            std_function = std_flat.reshape((n_prediction_dates, *y_grid.shape))
            std_function = np.ma.masked_invalid(std_function)
            to_netcdf(current_save_path, prediction_dates, 
                        y_grid[:,0], x_grid[0], 
                        mean_function, std_function, 
                        epsg=grid_epsg)
            print('Saved successfully')
            # except Exception as e:
            #     print(f'{target_date.strftime("%Y-%m-%d")} | {fish}, {gear} - Num data points: {len(df)}')
            #     print(e)
            #     continue
