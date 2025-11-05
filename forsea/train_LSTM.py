from typing import Optional

import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.dataloader import ForSeaDataset
from models.GaussianFilterLSTM import LSTMAutoEncoder


def train(
    config: dict,
    experiment: str = "/Experiments/Forsea",
    run_name: Optional[str] = None,
):
    gpu = torch.device("cuda:0")
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

    out_dim = len(config["fish_types"])
    model = LSTMAutoEncoder(
        (out_dim, *dataset.ocean_data_shape[3:]),
        land_mask=dataset.land_mask,
        **config["model_params"]
    ).cuda()

    # Loss
    if config["criterion"] == "mse":
        criterion = nn.MSELoss()
    elif config["criterion"] == "cross_entropy":
        criterion = nn.CrossEntropyLoss()
    elif config["criterion"] == "cross_entropy_with_logits":
        criterion = nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Invalid criterion: {config['criterion']}")

    if config["optimizer"] == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=config["lr"])
    elif config["optimizer"] == "adam":
        optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    else:
        raise ValueError(f"Invalid optimizer: {config['optimizer']}")

    batch_accumulation = config["batch_accumulation"]

    # MLFlow
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_param("epochs", config["epochs"])
        mlflow.log_param("lr", config["lr"])
        mlflow.log_param("criterion", config["criterion"])
        mlflow.log_param("optimizer", config["optimizer"])
        mlflow.log_param("batch_accumulation", config["batch_accumulation"])
        mlflow.log_param("mode", config["mode"])
        mlflow.log_param("thresholds", config["thresholds"])
        mlflow.log_param("log_target", config["log_target"])
        mlflow.log_param("fish_types", config["fish_types"])
        mlflow.log_param("gear_types", config["gear_types"])
        mlflow.log_param("sequence_len", config["sequence_len"])
        mlflow.log_params(config["model_params"])

        log_step = 0
        log_period = 200
        for epoch in range(config["epochs"]):
            model.train()
            optimizer.zero_grad()
            running_loss = 0.0
            train_batches = 0
            train_samples = 0
            for batch_idx, data in enumerate(dataset.train_data()):
                ocean_input, route_input, target = data
                if len(route_input) == 0:
                    continue
                
                outputs = model(ocean_input, route_input)
                loss = criterion(outputs, target)
                loss.backward()
                # running_loss += loss.item()
                # train_samples += len(outputs)
                
                # for output_step, target_step in zip(outputs, target):
                #     loss = criterion(output_step, target_step)
                #     loss = loss / config["sequence_len"]
                #     loss.backward(retain_graph=True)
                    
                #     running_loss += loss.item()
                #     train_samples += len(output_step)
                # optimizer.step()
                # optimizer.zero_grad()
                if ((batch_idx + 1) % batch_accumulation == 0) or (
                    batch_idx == dataset.train_n - 1
                ):
                    optimizer.step()
                    optimizer.zero_grad()
                running_loss += loss.item()
                train_batches += 1
                train_samples += len(route_input)
                # print metrics
                if (train_batches + 1) % log_period == 0:
                    print(
                        f"[{epoch + 1:3d}, {train_batches+1:5d}] | loss: {running_loss / train_samples:.6f}"
                    )
                    mlflow.log_metric(
                        "train_loss", running_loss / train_samples, step=log_step
                    )
                    log_step += 1
                    running_loss = 0.0
                    train_samples = 0
            # Validation
            val_loss = 0.0
            val_samples = 0
            model.eval()
            with torch.no_grad():
                for _, data in enumerate(dataset.val_data()):
                    ocean_input, route_input, target = data
                    if len(route_input) == 0:
                        continue
                                    
                    # outputs = model(ocean_input, route_input)
                    # for output_step, target_step in zip(outputs, target):
                    #     loss = criterion(output_step, target_step)
                    #     val_loss += loss.item() / config["sequence_len"]
                    #     val_samples += len(output_step)

                    outputs = model(ocean_input, route_input)
                    loss = criterion(outputs, target)
                    val_loss += loss.item() / batch_accumulation
                    val_samples += len(route_input)

            print(
                f"[{epoch + 1:3d}, {train_batches:5d}] | train loss: {running_loss / train_samples:.6f} | val loss: {val_loss / val_samples:.6f}"
            )
            mlflow.log_metric("val_loss", val_loss / val_samples, step=log_step)
    return model


if __name__ == "__main__":
    config = {
        "lr": 0.01,
        "epochs": 100,
        "batch_accumulation": 32,
        "fish_types": ["Torsk", "Sei", "Hyse"],
        "gear_types": ["Trål"],
        "mode": "regression",
        "sequence_len": 30,
        "log_target": False,
        "criterion": "cross_entropy_with_logits",
        "optimizer": "adam",
        "model_params": {"filters": (32, 64, 128, 256)},
        "ocean_data_path": "../data/copernicus/datasets/norway.nc",
        "route_data_path": "../data/VMS_DCA_joined/catch_routes.parquet",
    }

    train(config, run_name="Gaussian_filter_LSTM")
