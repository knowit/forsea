from typing import Optional

import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.dataloader import ForSeaDataset
from models.EncoderDecoder import ForseaAutoEncoder


def train(config: dict, experiment: str='/Experiments/Forsea', run_name: Optional[str]=None):
    gpu = torch.device('cuda:0')

    dataset = ForSeaDataset(config['ocean_data_path'], config['route_data_path'], log_target=config['log_target'], batched=True)

    model = ForseaAutoEncoder(
        ocean_input_shape=dataset.ocean_data_shape, 
        route_input_size=dataset.num_route_features, 
        output_size=dataset.target_size, 
        **config['model_params']
    ).cuda()

    # Loss
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    # MLFlow
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_param('epochs', config['epochs'])
        mlflow.log_param('lr', config['lr'])
        mlflow.log_param('log_target', config['log_target'])
        mlflow.log_params(config['model_params'])

        model.train()
        print_period = 500
        for epoch in range(config['epochs']):
            running_loss = 0.0
            num_batches = 0 
            for i, data in enumerate(dataset):
                (ocean_input, route_input), roundweight = data
                if len(route_input) == 0: continue

                optimizer.zero_grad()
                outputs = model(ocean_input, route_input)
                loss = criterion(outputs, roundweight)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                num_batches += 1
                # print metrics
                if num_batches % print_period == 0:
                    print(f'[{epoch + 1}, {num_batches:5d}] | loss: {running_loss / num_batches:.3f}')
            print(f'[{epoch + 1}, {num_batches:5d}] | loss: {running_loss / num_batches:.3f}')
            mlflow.log_metric('train_loss', running_loss / num_batches, step=epoch+1)
            

if __name__ == '__main__':
    
    config = {
        'lr': 0.001,
        'epochs': 10,
        'log_target': True,
        'model_params': {
            'filters': (16, 16, 16),
            'kernel_size': (3,3),
            'hidden_layers': (128,),
            'output_activation': None
        },
        'ocean_data_path': './data/copernicus/datasets/ocean_data.nc',
        'route_data_path': './data/VMS_DCA_joined/cod_trawl.parquet'
    }

    train(config)