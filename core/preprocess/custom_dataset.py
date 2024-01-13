from .origin_dataset import Dataset

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.preprocessing import OneHotEncoder
from .utils import merge_dataset


@dataclass(kw_only=True)
class CustomDataset(Dataset):
  drop_cols_after_merge: list
  fill_num_strategy: callable
  x_scaler: BaseEstimator = None
  y_scaler: BaseEstimator = None

  def _scale_features(self, df: pd.DataFrame):
    # if self.x_scaler == None:
    #   return df
    # if self.y_scaler == None:
    #   return df
    idx = df.index
    
    y_exists = False
    y_df = None
    X_df = df
    
    if self.target_col in df.columns:
      y_exists = True
      y_df = df[[self.target_col]].copy()
      X_df = df.drop(columns=[self.target_col])
    
    if self.x_scaler != None:
      self.x_scaler.fit(X_df)
      X_df = pd.DataFrame(
        self.x_scaler.transform(X_df).astype(dtype=np.float32),
        columns=self.x_scaler.get_feature_names_out()
      ).reset_index(drop=True)
    
    if self.y_scaler != None and y_exists:
      self.y_scaler.fit(y_df)
      y_df = pd.DataFrame(
        self.y_scaler.transform(y_df).astype(dtype=np.float32),
        columns=self.y_scaler.get_feature_names_out()
      ).reset_index(drop=True)
    
    if not y_exists: return X_df
    else:
      new_df = pd.concat([y_df, X_df], axis=1)
      new_df.set_index(idx, inplace=True)
      return new_df

  def _base_preprocess(self, X_df: pd.DataFrame):
    # Numeric
    df_num = X_df.select_dtypes(include=["number"])
    # core.utils.preprocess_tools.py
    df_num = self.fill_num_strategy(df_num)
    df_num.reset_index(drop=True, inplace=True)
    
    drop_block_id_len = df_num[df_num['data_block_id'] <= 1].shape[0]
    df_num.drop(columns=['data_block_id'], inplace=True)
    if self.x_scaler is not None or self.y_scaler is not None:
      df_num = pd.DataFrame(self._scale_features(df_num), columns=df_num.columns)

    # Categorical
    df_cat = X_df.select_dtypes(exclude=["number"])
    # enc = OneHotEncoder(
    #   dtype=np.float32,
    #   sparse_output=False,
    #   drop="if_binary",
    #   handle_unknown="ignore",
    # )
    # enc.fit(df_cat)
    # df_cat = pd.DataFrame(
    #   enc.transform(df_cat), columns=enc.get_feature_names_out()
    # ).reset_index(drop=True)

    new_df = pd.concat([df_num, df_cat], axis=1)
    new_df.set_index(self.index_col, inplace=True)
    new_df = new_df.iloc[drop_block_id_len:,:]
    return new_df

  def preprocess(self):
    """
    Returns:
        trn_X, tst_X, y_scaler
    """
    trn_df, tst_df, add_data = self._get_dataset()
    
    if self.use_arima == False:
      # Merge datasets
      trn_df, tst_df = merge_dataset(trn_df, tst_df, add_data)
      
      # Drop Columns after merge
      trn_df.drop(self.drop_cols_after_merge, axis=1, errors="ignore", inplace=True)
      tst_df.drop(self.drop_cols_after_merge, axis=1, errors="ignore", inplace=True)
      
      # X Features
      trn_df = self._base_preprocess(trn_df)
      tst_df = self._base_preprocess(tst_df)
    else:
      trn_df.fillna(method='ffill', inplace=True)
      if self.y_scaler != None:
        y_df = trn_df[[self.target_col]].copy()
        self.y_scaler.fit(y_df)
        y_df = pd.DataFrame(
          self.y_scaler.transform(y_df).astype(dtype=np.float32),
          columns=self.y_scaler.get_feature_names_out()
        )
        trn_df[self.target_col] = y_df[self.target_col].values
      trn_df.set_index(self.index_col, inplace=True)
      tst_df.set_index(self.index_col, inplace=True)

    print(trn_df.shape)
    print(tst_df.shape)
    return trn_df, tst_df, self.y_scaler
