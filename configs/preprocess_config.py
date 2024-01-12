from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from core.utils import get_root_path, fill_num_strategy


ROOT_PATH = get_root_path()

TARGET_COL = "target"
INDEX_COL = "datetime"
TST_INDEX_COL = "prediction_datetime"


### ROW FILTERING
# 데이터에서 아래의 조건에 맞춰 row 를 filtering 합니다.
# e.g. predict_unit_id 컬럼에서 값이 0 인 rows
ROW_QUERY = {
  "prediction_unit_id": 0,
  "is_consumption": 0
}

### DROP_X_COLS 와 USE_X_COLS 중 하나만 적용됩니다.
# USE_DROP_COLS 의 값에 따라서 결정됩니다.
# DROP_X_COLS 와 USE_X_COLS 중 TARGET_COL 과 INDEX_COL 이 포함되지 않도록 주의해주세요.
# use_arima 의 값이 True 일 경우, 사용되지 않습니다.
DROP_X_COLS = ["currently_scored"]
USE_X_COLS = []

USE_DROP_COLS = True
USE_ARIMA = False

DROP_COLS_AFTER_MERGE = [
  "county", "is_business", "is_consumption", "product_type", # "data_block_id", 
  "row_id", "date", "date_client", "forecast_date_electricity",
  "origin_date_electricity", "forecast_date_gas", "origin_date_gas", "prediction_unit_id",
  # "year", "quarter", "month", "week",  "day", "hour",
  # "day_of_year", "day_of_month", "day_of_week", "eic_count_client",
  # "installed_capacity_client", "temperature_h_mean", 
  # "dewpoint_h_mean",
  # "rain_h_mean", "snowfall_h_mean", "surface_pressure_h_mean",
  # "cloudcover_total_h_mean", "cloudcover_low_h_mean",
  # "cloudcover_mid_h_mean", "cloudcover_high_h_mean",
  # "windspeed_10m_h_mean", "winddirection_10m_h_mean",
  # "shortwave_radiation_h_mean", "direct_solar_radiation_h_mean",
  # "diffuse_radiation_h_mean", "hour_h", "hours_ahead_f_mean",
  # "temperature_f_mean", "dewpoint_f_mean", "cloudcover_high_f_mean",
  # "cloudcover_low_f_mean", "cloudcover_mid_f_mean",
  # "cloudcover_total_f_mean", "10_metre_u_wind_component_f_mean",
  # "10_metre_v_wind_component_f_mean", "direct_solar_radiation_f_mean",
  # "surface_solar_radiation_downwards_f_mean", "snowfall_f_mean",
  # "total_precipitation_f_mean", "euros_per_mwh_electricity",
  # "lowest_price_per_mwh_gas", "highest_price_per_mwh_gas",
  # "mean_price_per_mwh_gas", "country_holiday"
]

# choose one: ['min', 'mean', 'max', 'custom_mean']
FILL_NUM_STRATEGY = fill_num_strategy("custom_mean")


# Custom scaler 의 경우, class 를 생성하여 사용해주세요.
SCALER = {
  "standard": StandardScaler,
  "minmax": MinMaxScaler,
  "maxabs": MaxAbsScaler,
  "robust": RobustScaler,
}
SELECTED_X_SCALER = None
SELECTED_Y_SCALER = None
# SELECTED_X_SCALER = SCALER["minmax"]() # 사용을 원치 않을 경우, None 값을 넣어주세요.
# SELECTED_Y_SCALER = SCALER["minmax"]() # 사용을 원치 않을 경우, None 값을 넣어주세요.

config = {
  # origin data path
  "input_data": {
    "train_csv": f"{ROOT_PATH}/data/train.csv",
    "test_csv": f"{ROOT_PATH}/data/test.csv"
  },
  # save preprocess results
  "output_data": {
    "train_csv": f"{ROOT_PATH}/features/train_X.csv",
    "test_csv": f"{ROOT_PATH}/features/test_X.csv",
    "y_scaler_save": f"{ROOT_PATH}/features/y_scaler.save",
  },
  "add_data": {
    "train": {
      "client": f"{ROOT_PATH}/data/train/client.csv",
      "historical_weather": f"{ROOT_PATH}/data/train/historical_weather.csv",
      "forecast_weather": f"{ROOT_PATH}/data/train/forecast_weather.csv",
      "electricity_prices": f"{ROOT_PATH}/data/train/electricity_prices.csv",
      "gas_prices": f"{ROOT_PATH}/data/train/gas_prices.csv",
    },
    "test": {
      "client": f"{ROOT_PATH}/data/test/client.csv",
      "historical_weather": f"{ROOT_PATH}/data/test/historical_weather.csv",
      "forecast_weather": f"{ROOT_PATH}/data/test/forecast_weather.csv",
      "electricity_prices": f"{ROOT_PATH}/data/test/electricity_prices.csv",
      "gas_prices": f"{ROOT_PATH}/data/test/gas_prices.csv",
    },
    "county_mapper": {"county_mapper": f"{ROOT_PATH}/data/county_lon_lats.csv"}
  },
  "options": {
    # origin dataset
    "index_col": INDEX_COL,
    "tst_index_col": TST_INDEX_COL,
    "target_col": TARGET_COL,
    "use_arima": USE_ARIMA,
    "row_query": ROW_QUERY,
    "use_drop_cols": USE_DROP_COLS,
    "drop_cols": DROP_X_COLS,
    "use_cols": USE_X_COLS,
    # custom dataset
    "drop_cols_after_merge": DROP_COLS_AFTER_MERGE,
    "fill_num_strategy": FILL_NUM_STRATEGY,
    "x_scaler": SELECTED_X_SCALER,
    "y_scaler": SELECTED_Y_SCALER,
  },
}
