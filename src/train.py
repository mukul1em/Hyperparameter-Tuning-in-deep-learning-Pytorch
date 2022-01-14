import torch
from torch.serialization import save
import utils
import optuna
import pandas as pd
import numpy as np


Device = 'cuda'
epochs = 100

def run_training(params, fold, save_model=False):
    df = pd.read_csv("input/train_features.csv")
    df = df.drop(["cp_type","cp_time", "cp_dose"], axis=1)

    targets_df = pd.read_csv("input/train_targets_folds.csv")

    features_columns = df.drop("sig_id", axis=1).columns
    target_columns = targets_df.drop(["sig_id","kfold"],axis=1).columns

    df = df.merge(targets_df, on ="sig_id", how='left')


    train_df = df[df.kfold != fold].reset_index(drop=True)
    valid_df = df[df.kfold == fold].reset_index(drop=True)

    xtrain = train_df[features_columns].to_numpy()
    ytrain  = train_df[target_columns].to_numpy()

    xvalid = valid_df[features_columns].to_numpy()
    yvalid  = valid_df[target_columns].to_numpy()

    train_dataset = utils.MoaDataset(features=xtrain, targets=ytrain)
    valid_dataset = utils.MoaDataset(features=xvalid, targets=yvalid)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1024, num_workers=8, shuffle=True
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=1024, num_workers=8
    )


    model = utils.Model(
        nfeatures=xtrain.shape[1], 
        ntargets=ytrain.shape[1], 
        nlayers=params["num_layers"], 
        hidden_size=params["hidden_size"], 
        dropout=params["dropout"]
    )

    model.to(Device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    eng = utils.Engine(model, optimizer, device=Device)

    best_loss = np.inf
    early_stopping_iter = 10
    early_stopping_counter = 0

    for epoch in range(epochs):
        train_loss  = eng.train_fn(train_loader)
        valid_loss  = eng.evaluate(valid_loader)
        print(f"{fold}, {epoch},{train_loss},{valid_loss}")
        if valid_loss < best_loss:
            best_loss = valid_loss
            if save_model:
                torch.save(model.state_dict(), f"model_{fold}.bin")
            else:
                early_stopping_counter += 1
            #if we are not improving in 10 iterations than stop the training 
            if early_stopping_counter > early_stopping_iter:
                break
    return best_loss



def objective(trial):
    params = {
        "num_layers":trial.suggest_int("num_layers",1,7),
        "hidden_size":trial.suggest_int("hidden_size",16,2048),
        "dropout":trial.suggest_uniform("dropout", 0.1, 0.7),
        "learning_rate":trial.suggest_loguniform("learning_rate", 1e-6, 1e-3)
        }

    all_losses = []
    #5 folds
    for f_ in range(5):
        temp_loss = run_training(params,fold=0, save_model=False)
        all_losses.append(temp_loss)
    return np.mean(all_losses)
   
if __name__  == "__main__":
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20)
    print("best trial: ")
    trial_ = study.best_trial
    print(trial_.values)
    print(trial_.params)

    scores = 0

    #folds
    for j in range(5):
        scr = run_training(j, trial_.params, save_model=True)
        scores+=scr

  

