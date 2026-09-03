# export cuda_visible_devices=2,3
import argparse
import datetime
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.optim.lr_scheduler import StepLR

from models.HDAAGT import *
from train_test import *
from utilz.utils import *


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the main script with a config file.")
    parser.add_argument("-c", "--config", type=str, help="Path to the config.yaml file")
    args = parser.parse_args()
    torch.cuda.empty_cache()

    if args.config is None:
        args.config = "configs/config.yaml"
        print(f"Using the '{args.config}' file")

    with open(args.config) as file:
        config = yaml.safe_load(file)

    seed = int(config.get('seed', 42))
    config['seed'] = seed
    set_seed(seed)

    cwd = os.getcwd()
    requested_device = str(config.get('device', 'cuda:0'))
    if requested_device.startswith('cuda') and torch.cuda.is_available():
        requested_index = 0
        if ':' in requested_device:
            try:
                requested_index = int(requested_device.rsplit(':', 1)[1])
            except ValueError:
                requested_index = 0
        if requested_index < torch.cuda.device_count():
            device = torch.device(f'cuda:{requested_index}')
        else:
            device = torch.device('cuda:0')
            print(
                f"Requested device {requested_device} is unavailable; "
                "falling back to cuda:0"
            )
    else:
        device = torch.device('cpu')

    ct = datetime.datetime.now().strftime("%m-%d-%H-%M")
    savelog(f"Total  # of GPUs {torch.cuda.device_count()}", ct)
    savelog(f"Using {device} device", ct)
    savelog(f' Model description: {config["description"]}', ct=ct)

    ZoneConf = Zoneconf(config['Zoneconf_path'])
    config['NFeatures'] = len(config["Columns_to_keep"])
    config["input_size"] = config["NFeatures"] + 9 + 8
    config["output_size"] = len(config['xy_indx'])
    config['device'] = device
    config['ct'] = ct
    log_code = f"Learning rate: {config['learning_rate']}, Hidden size: {config['hidden_size']}, Batch size: {config['batch_size']}"
    if config['verbal']:
        for arg in config.items():
            savelog(f"{arg[0]} = {arg[1]}", ct)
    config['sos'] = torch.cat((torch.tensor(config['sos']), torch.zeros(17)), dim=0).repeat(config['Nnodes'], 1).to(device)
    config['eos'] = torch.cat((torch.tensor(config['eos']), torch.zeros(17)), dim=0).repeat(config['Nnodes'], 1).to(device)
    best_path = os.path.join(cwd, 'Pickled', f'best_trained_model_{ct}.pth')
    loss_path = os.path.join(cwd, 'Pickled', f'epoch_losses_{ct}.pt')

    os.makedirs(os.path.join(cwd, 'Pickled'), exist_ok=True)

    Scenetr = Scenes(config)
    Scenetst = Scenes(config)
    Sceneval = Scenes(config)
    savelog(f'Starting at:  {datetime.datetime.now().strftime("%H-%M-%S")}', ct)
    if not config['generate_data']:
        savelog("Loading data ...", ct)
        Scenetr.load_class('Pickled', cmnt='Train')
        Scenetst.load_class('Pickled', cmnt='Test')
        Sceneval.load_class('Pickled', cmnt='Validation')
    else:
        savelog("Loading CSV file ...", ct)
        Traffic_data = loadcsv(config['detection_path'], config['Headers'])
        Scenetr, Scenetst, Sceneval = Scene_Process(Scenetr, Scenetst, Sceneval, Traffic_data, config)
        print(f"Augmenting the data {config['noise_multiply']} times")
        Scenetr.addnoise(config['noise_multiply'], config['noise_amp'], config['noise_ratio'])
        Scenetr.save_class('Pickled', cmnt='Train')
        Scenetst.save_class('Pickled', cmnt='Test')
        Sceneval.save_class('Pickled', cmnt='Validation')

    savelog(f"Train size: {len(Scenetr)}, Test size: {len(Scenetst)}, Validation size: {len(Sceneval)}", ct)

    train_loader, test_loader, val_loader = DataLoader_Scene(Scenetr, Scenetst, Sceneval, config['batch_size'])

    model = HDAAGT(config).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
    scheduler = StepLR(optimizer, step_size=config['schd_stepzise'], gamma=config['gamma'])
    if not config['Load_Model']:
        savelog("Creating model from scratch", ct)
    if config['Load_Model']:
        savelog("Loading model from the checkpoint", ct)
        model.load_state_dict(torch.load(config['Load_Model_Path'], map_location=device))

    Best_Model = model
    if config['Train']:
        savelog(f"The number of learnable parameter is {count_parameters(model=model)} !", ct)
        print(f"Learning rate: {config['learning_rate']}, Hidden size: {config['hidden_size']}, Batch size: {config['batch_size']}")
        Best_Model, train_loss, trainADE, trainFDE = train_model(model, optimizer, criterion, scheduler, train_loader, test_loader, config)
        savelog(f"{log_code} with Avg loss of {train_loss[-1]}, ADE of {trainADE[-1]}, FDE of {trainFDE[-1]}", f"summary {ct}")
        savelog(f"Saving result of {log_code}", ct)
        torch.save(Best_Model.state_dict(), best_path)
        torch.save(train_loss, loss_path)
        savelog(f"Best checkpoint saved to {best_path}", ct)
        savelog(f"Training finished for {ct}!", ct)

    eval_model = Best_Model if config['Train'] else model
    if config['Test']:
        Topk_Selected_words, ADE, FDE = test_model(eval_model, test_loader, config)
        px_to_m = 0.75 / 9.0
        savelog(
            f"Average Displacement Error: {ADE:.3f} px ({ADE * px_to_m:.3f} m), "
            f"Final Displacement Error: {FDE:.3f} px ({FDE * px_to_m:.3f} m)",
            ct,
        )
