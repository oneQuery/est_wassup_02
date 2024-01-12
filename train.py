from core.dataloader import TimeseriesDataset
from core.metrics import mae, mape, r2_score
from core.trains import nn_train, nn_predict
from core.utils import (
    create_path_if_not_exists,
    save_params_json,
    get_args_parser,
    combine_paths,
    SaveBestModel
)
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import joblib
import random


# train & test together
def main(cfg):
    ## load datasets
    # get preprocessed csv files
    input_data = cfg.get("input_data")
    train_data = pd.read_csv(input_data.get('train_csv'), index_col=0)
    train_data.index = pd.to_datetime(train_data.index) # change index data type
    
    model_params = cfg.get("model_params")
    window_size = model_params.get("input_dim") # 24
    prediction_size = model_params.get("output_dim")
    test_size = cfg.get("test_size") # 96


    ## split dataset
    # trn & val data
    trn_ds = train_data[:-test_size*2].to_numpy(dtype=np.float32)
    val_ds = train_data[-window_size-test_size*2:-test_size].to_numpy(dtype=np.float32) # 96 step + 24 step = 120 steps
    
    # test data
    tst_ds_y = train_data[-test_size:]['target'].copy()
    tst_df = train_data[-window_size-test_size:].copy()
    tst_df['target'][-test_size:] = np.nan
    tst_ds = tst_df.to_numpy(dtype=np.float32) # 96 step + 24 step = 120 steps
    
    print("trn.shape:", trn_ds.shape)
    print("val.shape:", val_ds.shape)
    print("tst.shape:", tst_ds.shape)
    
    
    ## generate dataloader
    train_params = cfg.get("train_params")
    ds_params = train_params.get("dataset_params")
    dl_params = train_params.get("data_loader_params")
    
    # train
    trn_ds = TimeseriesDataset(trn_ds, **ds_params)
    trn_dl = DataLoader(trn_ds, **dl_params)
    
    # validation
    val_ds = TimeseriesDataset(val_ds, **ds_params)
    val_dl = DataLoader(val_ds, shuffle=False, batch_size=len(val_ds))
    
    # test
    tst_ds = TimeseriesDataset(tst_ds, **ds_params)
    tst_dl = DataLoader(tst_ds, shuffle=False, batch_size=1)
    
    
    ## generate model
    # device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps")

    # create model
    Model = cfg.get("model")
    model_params = cfg.get("model_params")
    n_featues = train_data.shape[1]
    model = Model(**{**model_params, "input_channel": n_featues}).to(device)
    
    # optimizer
    Optim = train_params.get("optim")
    optim_params = train_params.get("optim_params")
    optimizer = Optim(model.parameters(), **optim_params)
    
    # learning rate scheduler
    LRScheduler = train_params.get("scheduler_cls") 
    scheduler_params = train_params.get("scheduler_params")
    use_scheduler = train_params.get("use_scheduler")
    scheduler = None
    if use_scheduler:
        scheduler = LRScheduler(optimizer, **scheduler_params)
    
    # early stop
    use_early_stop = train_params.get("use_early_stop")
    early_stop_params = train_params.get("early_stop_params")

    loss = train_params.get("loss")
    metric = train_params.get("metric")
    epochs = train_params.get("epochs")
    
    
    ## best model saver initializer
    output_data = cfg.get("output_data")
    root_dir = create_path_if_not_exists(output_data.get("root_dir"), remove_filename=False)
    best_model_save_path = combine_paths(root_dir, output_data.get("output_best_train"))
    best_model_saver = SaveBestModel(best_model_save_path)
    
    
    ## TRAIN
    trn_loss_lst, val_loss_lst = nn_train(epochs, trn_dl, val_dl, model, loss, metric, optimizer, device, scheduler, use_early_stop, early_stop_params, best_model_saver)


    ## save pre-trained weight
    final_model_save_path = combine_paths(root_dir, output_data.get('output_train'))
    torch.save(model.state_dict(), final_model_save_path)
    save_params_json(combine_paths(root_dir, output_data.get("json_path")), cfg)


    ## load models
    # final model
    model = Model(**{**model_params, "input_channel": n_featues}).to(device)
    model.load_state_dict(torch.load(final_model_save_path))
    model.eval()
    # best model
    best_model = Model(**{**model_params, "input_channel": n_featues}).to(device)
    best_model.load_state_dict(torch.load(best_model_save_path))
    best_model.eval()
    
    
    ## Prediction
    # retrieve from test dataloader
    pred = nn_predict(tst_dl, model, test_size, prediction_size, window_size, device)
    pred_best_model = nn_predict(tst_dl, best_model, test_size, prediction_size, window_size, device)

    # inverse scaling
    y_scaler = None
    try:
        y_scaler = joblib.load(input_data.get("y_scaler_save"))
    except:
        pass
    
    if y_scaler != None:
        y = y_scaler.inverse_transform(pd.DataFrame(tst_ds_y))
        p = y_scaler.inverse_transform([pred])
        p_best_model = y_scaler.inverse_transform([pred_best_model])
    else:
        y = pd.DataFrame(tst_ds_y).to_numpy()
        p = np.array([pred])
        p_best_model = np.array([pred_best_model])

    y = np.concatenate([y[:,0], y[-1,1:]]) # true 
    p = np.concatenate([p[:,0], p[-1,1:]]) # prediction
    p_best_model = np.concatenate([p_best_model[:,0], p_best_model[-1,1:]]) # prediction
    print("y-shape:", y.shape)
    print("p-shape:", p.shape)
    print("b-shape:", p_best_model.shape)

    pred_df = pd.DataFrame({'prediction': p})
    pred_df.to_csv(combine_paths(root_dir, output_data.get("output_pred")))


    ### Visualization with Results ###
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))
    
    # Plotting the losses
    ax1.plot(range(1,len(trn_loss_lst)+1), trn_loss_lst, label='Training Loss')
    ax1.plot(range(1,len(val_loss_lst)+1), val_loss_lst, label='Validation Loss')
    ax1.set_title(f"{Model}, last train loss: {trn_loss_lst[-1]:.6f}, last valiadtion loss: {val_loss_lst[-1]:.6f}")
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Visualize results - true, pred with metrics  
    ax2.plot(tst_ds_y.index, y, label="True")
    ax2.plot(tst_ds_y.index, p, label="Prediction")
    ax2.set_title(f"{Model}, MAPE:{mape(p,y):.4f}, MAE:{mae(p,y):.4f}, R2:{r2_score(p,y):.4f}")
    ax2.legend()
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")
    date_format = mdates.DateFormatter('%Y-%m-%d %H')
    ax2.xaxis.set_major_formatter(date_format)
    plt.xticks(tst_ds_y.index[::3])
    
    # Visualize results - true, pred with metrics  
    ax3.plot(tst_ds_y.index, y, label="True")
    ax3.plot(tst_ds_y.index, p_best_model, label="Prediction")
    ax3.set_title(f"Best model: {Model}, MAPE:{mape(p_best_model,y):.4f}, MAE:{mae(p_best_model,y):.4f}, R2:{r2_score(p_best_model,y):.4f}")
    ax3.legend()
    plt.setp(ax3.get_xticklabels(), rotation=45, ha="right")
    date_format = mdates.DateFormatter('%Y-%m-%d %H')
    ax3.xaxis.set_major_formatter(date_format)
    plt.xticks(tst_ds_y.index[::3])
    
    plt.tight_layout()
    plt.savefig(combine_paths(root_dir, output_data.get("plot_img_path"))) # save as png
    
    
if __name__ == "__main__":
    seed = 2023
    deterministic = True

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    args = get_args_parser(config_type='train').parse_args()
    config = {}
    exec(open(args.config, encoding="utf-8").read())
    main(config)
    print("finished! Please check the output folder.")