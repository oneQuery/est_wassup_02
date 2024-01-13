from tqdm.auto import trange
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import Optimizer
from .tst_validate import validate_one_epoch
from ..utils import SaveBestModel
from tqdm import tqdm


# train function for one epoch
def train_one_epoch(dataloader, model, loss_func, optimizer, device, scheduler):
  model.train()
  total_loss = 0. 
  for x, y in tqdm(dataloader): 
    x, y = x.to(device), y.to(device) 
    pred = model(x)
    loss = loss_func(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    total_loss += loss.item()*len(y)
  trn_loss = total_loss/len(dataloader.dataset)
  if scheduler != None: scheduler.step()
  return trn_loss

def tst_train(
  epochs:int, trn_dl: DataLoader, val_dl: DataLoader, model: nn.Module,
  loss: callable, metric: callable, optimizer: Optimizer, device,
  scheduler, use_early_stop: bool, early_stop_params: dict, save_best_model: SaveBestModel
):
  best_loss = float("inf")
  patience_limit = early_stop_params.get("patience_limit")
  patience_check = 0
  
  trn_loss_lst = []
  val_loss_lst = []

  # training and progress bar display
  pbar = trange(epochs)
  for i in pbar: # loop
    # FIXME
    trn_loss = train_one_epoch(trn_dl, model, loss, optimizer, device, scheduler)
    val_loss, val_metric = validate_one_epoch(val_dl, model, loss, metric, device)
    trn_loss_lst.append(trn_loss)
    val_loss_lst.append(val_loss)
    pbar.set_postfix({'trn_mse': trn_loss, 'val_mse': val_loss, 'val_mae': val_metric})
    print(f"trn_mse: {trn_loss}, val_mse: {val_loss}, val_mae: {val_metric}")
    
  
    # save best model
    save_best_model(val_loss, model)
  
    # EARLY STOP
    if use_early_stop and val_loss > best_loss:
      patience_check += 1
      if patience_check > patience_limit: 
        print(f"early stopping: epoch {i}")
        break
    else:
      best_loss = val_loss
      patience_check = 0
  
  return trn_loss_lst, val_loss_lst
