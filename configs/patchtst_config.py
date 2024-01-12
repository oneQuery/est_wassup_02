import torch
import torch.nn.functional as F
from torch import optim
from torch.nn import ReLU

from core.utils import get_root_path
from models import PatchTST

ROOT_PATH = get_root_path()
EXPERIMENT_NAME = "patch_test_2_w_scheduler"

TEST_SIZE = 24

TARGET_COL = "target"

# train params
EPOCHS = 40
LEARNING_RATE = 0.00001 # e-4
# BATCH_SIZE = 32
LOSS_FUNC = F.mse_loss
OPTIM = torch.optim.Adam
METRIC = F.l1_loss

# model params
# MODEL = PatchTST
PRED_SIZE = 24
# PATCH_LENGTH = 16 # 논문 수치
# N_PATCHES = 64 # 논문 수치: 64 or 96
# # WINDOW_SIZE = int(PATCH_LENGTH*N_PATCHES/2) # 512 약 21일
# MODEL_DIM = 128 # HIDDEN_DIM 같은 것
# NUM_HEADS = 8 # 논문 수치로 일단 고정
# NUM_LAYERS = 3 # 논문 수치로 일단 고정
# DIM_FEED_FORWARD = 256
# OUTPUT_FUNC = F.sigmoid # Y scaler 에 따라서 변경되어야 함
# # y scaler 가 None 인 경우, lambda x: x 이렇게 넣어주면 됨.

# model params
MODEL = PatchTST

BATCH_SIZE = 512

PATCH_LENGTH = 2 # 논문 수치
N_PATCHES = 8 # 논문 수치: 64 or 96
# WINDOW_SIZE = int(PATCH_LENGTH*N_PATCHES/2) # 512 약 21일
MODEL_DIM = 32 # HIDDEN_DIM 같은 것
NUM_HEADS = 2 # 논문 수치로 일단 고정
NUM_LAYERS = 1 # 논문 수치로 일단 고정
DIM_FEED_FORWARD = 32
OUTPUT_FUNC = F.sigmoid # Y scaler 에 따라서 변경되어야 함
# y scaler 가 None 인 경우, lambda x: x 이렇게 넣어주면 됨.

USE_SCHEDULER = True
SCHEDULER = optim.lr_scheduler.CosineAnnealingLR
SCHEDULER_PARAMS = {
    "T_max": 40,
    "eta_min": 0.000001, #e-5
}

# early stop
USE_EARLY_STOP = True
EARLY_STOP_PARAMS = {
    "patience_limit": 10 # 몇 번의 epoch까지 지켜볼지를 결정
}

config = {
    "input_data": {
        "train_csv": f"{ROOT_PATH}/features/train_X.csv",
        # "test_csv": f"{ROOT_PATH}/features/test_X.csv",
        "y_scaler_save": f"{ROOT_PATH}/features/y_scaler.save",
    },
    "output_data": {
        "root_dir": f"{ROOT_PATH}/output/{EXPERIMENT_NAME}",
        "output_train": "model.pth",
        "plot_img_path": "output.png",
        "json_path": "config.json",
        "output_pred": "pred.csv",
        "output_best_train": "best_model.pth"
    },
    "test_size": TEST_SIZE,
    "window_size": int(PATCH_LENGTH*N_PATCHES/2),
    "model": MODEL,
    "model_params": {
        "n_token": N_PATCHES,
        "input_dim": PATCH_LENGTH,  
        "model_dim": MODEL_DIM,  
        "num_heads": NUM_HEADS, 
        "num_layers": NUM_LAYERS,
        "dim_feedforward": DIM_FEED_FORWARD,
        "output_dim": PRED_SIZE,
        "output_func": OUTPUT_FUNC
    },
    "train_params": {
        "use_scheduler": USE_SCHEDULER,
        "scheduler_cls": SCHEDULER,
        "scheduler_params": SCHEDULER_PARAMS,
        "use_early_stop": USE_EARLY_STOP,
        "early_stop_params": EARLY_STOP_PARAMS,
        "loss": LOSS_FUNC,
        "optim": OPTIM,
        "metric": METRIC,
        "epochs": EPOCHS,
        "dataset_params": {
            "patch_length": PATCH_LENGTH,
            "n_patches": N_PATCHES,
            "prediction_length": PRED_SIZE
        },
        "data_loader_params": {
            "batch_size": BATCH_SIZE,
            "shuffle": True,
        },
        "optim_params": {
            "lr": LEARNING_RATE,
        },
    },
}
