# config

ROOT_DATA_PATH = '../data/origin/'
TEST_DATA_PATH = f'{ROOT_DATA_PATH}test/'
TRAIN_DATA_PATH = f'{ROOT_DATA_PATH}train/'

ROOT_FEATURE_PATH = '../data/features'

data_path_config = {
  'input': {
    'train': {
      'train': f'{TRAIN_DATA_PATH}train.csv',
      'client': f'{TRAIN_DATA_PATH}client.csv',
      'electricity': f'{TRAIN_DATA_PATH}electricity_prices.csv',
      'forecast_weather': f'{TRAIN_DATA_PATH}forecast_weather.csv',
      'gas': f'{TRAIN_DATA_PATH}gas_prices.csv',
      'historical_weather': f'{TRAIN_DATA_PATH}historical_weather.csv'
    },
    'test': {
      'test': f'{TEST_DATA_PATH}test.csv',
      'client': f'{TEST_DATA_PATH}client.csv',
      'electricity': f'{TEST_DATA_PATH}electricity_prices.csv',
      'forecast_weather': f'{TEST_DATA_PATH}forecast_weather.csv',
      'gas': f'{TEST_DATA_PATH}gas_prices.csv',
      'historical_weather': f'{TEST_DATA_PATH}historical_weather.csv'
    },
    # instead of weather_station_to_county_mapping.csv, use fabiendaniels-mapping-locations-and-county-codes
    # https://www.kaggle.com/datasets/michaelo/fabiendaniels-mapping-locations-and-county-codes/data
    'county_mapper': f'{ROOT_DATA_PATH}county_lon_lats.csv',
  },
  'output': {
    'train': f'{ROOT_FEATURE_PATH}/train.csv',
    'test': f'{ROOT_FEATURE_PATH}/test.csv'
  }
}


import pandas as pd


# 출처: https://www.kaggle.com/code/rafiko1/enefit-xgboost-starter#Data-processing
# 추후 수정 필요합니다.
# 우선 무지성으로 죄다 머지하는 코드입니다.
# 각 코드에 대한 이해는 같은 폴더 하위에 있는 merge_data.ipynb 확인하시기 바랍니다.
class FeatureProcessorClass():
  def __init__(self):         
    # Columns to join on for the different datasets
    self.weather_join = ['datetime', 'county', 'data_block_id']
    self.gas_join = ['data_block_id']
    self.electricity_join = ['datetime', 'data_block_id']
    self.client_join = ['county', 'is_business', 'product_type', 'data_block_id']
    
    # Columns of latitude & longitude
    self.lat_lon_columns = ['latitude', 'longitude']
    
    # Aggregate stats 
    self.agg_stats = ['mean'] #, 'min', 'max', 'std', 'median']
    
    # Categorical columns (specify for XGBoost)
    self.category_columns = ['county', 'is_business', 'product_type', 'is_consumption', 'data_block_id']

  def create_new_column_names(self, df, suffix, columns_no_change):
    '''Change column names by given suffix, keep columns_no_change, and return back the data'''
    df.columns = [
      col + suffix 
      if col not in columns_no_change
      else col
      for col in df.columns
    ]
    return df 

  def flatten_multi_index_columns(self, df):
    df.columns = ['_'.join([col for col in multi_col if len(col)>0]) 
                  for multi_col in df.columns]
    return df
  
  def create_data_features(self, data):
    '''📊Create features for main data (test or train) set📊'''
    # To datetime
    if 'prediction_datetime' in data.columns:
      data.rename(columns={'prediction_datetime': 'datetime'}, inplace=True)
    data['datetime'] = pd.to_datetime(data['datetime'])
    
    # Time period features
    data['date'] = data['datetime'].dt.normalize()
    data['year'] = data['datetime'].dt.year
    data['quarter'] = data['datetime'].dt.quarter
    data['month'] = data['datetime'].dt.month
    data['week'] = data['datetime'].dt.isocalendar().week
    data['hour'] = data['datetime'].dt.hour
    
    # Day features
    data['day_of_year'] = data['datetime'].dt.day_of_year
    data['day_of_month']  = data['datetime'].dt.day
    data['day_of_week'] = data['datetime'].dt.day_of_week
    return data

  def create_client_features(self, client):
    '''💼 Create client features 💼'''
    # Modify column names - specify suffix
    client = self.create_new_column_names(
      client, 
      suffix='_client',
      columns_no_change = self.client_join
      )       
    return client
  
  def create_historical_weather_features(self, historical_weather):
    '''⌛🌤️ Create historical weather features 🌤️⌛'''
    
    # To datetime
    historical_weather['datetime'] = pd.to_datetime(historical_weather['datetime'])
    
    # Add county
    historical_weather[self.lat_lon_columns] = historical_weather[self.lat_lon_columns].astype(float).round(1)
    historical_weather = historical_weather.merge(location, how = 'left', on = self.lat_lon_columns)

    # Modify column names - specify suffix
    historical_weather = self.create_new_column_names(
      historical_weather,
      suffix='_h',
      columns_no_change = self.lat_lon_columns + self.weather_join
    )
    
    # Group by & calculate aggregate stats 
    agg_columns = [col for col in historical_weather.columns if col not in self.lat_lon_columns + self.weather_join]
    agg_dict = {agg_col: self.agg_stats for agg_col in agg_columns}
    historical_weather = historical_weather.groupby(self.weather_join).agg(agg_dict).reset_index() 
    
    # Flatten the multi column aggregates
    historical_weather = self.flatten_multi_index_columns(historical_weather) 
    
    # Test set has 1 day offset for hour<11 and 2 day offset for hour>11
    historical_weather['hour_h'] = historical_weather['datetime'].dt.hour
    historical_weather['datetime'] = (
      historical_weather
      .apply(
        lambda x: 
          x['datetime'] + pd.DateOffset(1) 
          if x['hour_h']< 11 
          else x['datetime'] + pd.DateOffset(2),
        axis=1
      )
    )
    
    return historical_weather
  
  def create_forecast_weather_features(self, forecast_weather):
    '''🔮🌤️ Create forecast weather features 🌤️🔮'''
    
    # Rename column and drop
    forecast_weather = (
      forecast_weather
      .rename(columns = {'forecast_datetime': 'datetime'})
      .drop(columns = 'origin_datetime') # not needed
    )
    
    # To datetime
    forecast_weather['datetime'] = (
      pd.to_datetime(forecast_weather['datetime'])
      .dt
      .tz_localize(None)
      )

    # Add county
    forecast_weather[self.lat_lon_columns] = forecast_weather[self.lat_lon_columns].astype(float).round(1)
    forecast_weather = forecast_weather.merge(location, how = 'left', on = self.lat_lon_columns)
    
    # Modify column names - specify suffix
    forecast_weather = self.create_new_column_names(
      forecast_weather,
      suffix='_f',
      columns_no_change = self.lat_lon_columns + self.weather_join
    )
      
    # Group by & calculate aggregate stats 
    agg_columns = [col for col in forecast_weather.columns if col not in self.lat_lon_columns + self.weather_join]
    agg_dict = {agg_col: self.agg_stats for agg_col in agg_columns}
    forecast_weather = forecast_weather.groupby(self.weather_join).agg(agg_dict).reset_index() 
    
    # Flatten the multi column aggregates
    forecast_weather = self.flatten_multi_index_columns(forecast_weather)     
    return forecast_weather

  def create_electricity_features(self, electricity):
    '''⚡ Create electricity prices features ⚡'''
    # To datetime
    electricity['forecast_date'] = pd.to_datetime(electricity['forecast_date'])
    
    # Test set has 1 day offset
    electricity['datetime'] = electricity['forecast_date'] + pd.DateOffset(1)
    
    # Modify column names - specify suffix
    electricity = self.create_new_column_names(
      electricity, 
      suffix='_electricity',
      columns_no_change = self.electricity_join
    )             
    return electricity

  def create_gas_features(self, gas):
    '''⛽ Create gas prices features ⛽'''
    # Mean gas price
    gas['mean_price_per_mwh'] = (gas['lowest_price_per_mwh'] + gas['highest_price_per_mwh'])/2
    
    # Modify column names - specify suffix
    gas = self.create_new_column_names(
      gas, 
      suffix='_gas',
      columns_no_change = self.gas_join
    )       
    return gas
  
  def __call__(self, data, client, historical_weather, forecast_weather, electricity, gas):
    '''Processing of features from all datasets, merge together and return features for dataframe df '''
    # Create features for relevant dataset
    data = self.create_data_features(data)
    client = self.create_client_features(client)
    historical_weather = self.create_historical_weather_features(historical_weather)
    forecast_weather = self.create_forecast_weather_features(forecast_weather)
    electricity = self.create_electricity_features(electricity)
    gas = self.create_gas_features(gas)
    
    # 🔗 Merge all datasets into one df 🔗
    df = data.merge(client, how='left', on = self.client_join)
    df = df.merge(historical_weather, how='left', on = self.weather_join)
    df = df.merge(forecast_weather, how='left', on = self.weather_join)
    df = df.merge(electricity, how='left', on = self.electricity_join)
    df = df.merge(gas, how='left', on = self.gas_join)
    
    # Change columns to categorical for XGBoost
    df[self.category_columns] = df[self.category_columns].astype('category')
    return df

if __name__ == "__main__":
  train_data = {
    k: pd.read_csv(v) for k,v in data_path_config['input']['train'].items()
  }
  
  test_data = {
    k: pd.read_csv(v) for k,v in data_path_config['input']['test'].items()
  }
  
  location = pd.read_csv(data_path_config['input']['county_mapper'])
  
  
  FeatureProcessor = FeatureProcessorClass()

  df_train = FeatureProcessor(
    data = train_data['train'].copy(),
    client = train_data['client'].copy(),
    historical_weather = train_data['historical_weather'].copy(),
    forecast_weather = train_data['forecast_weather'].copy(),
    electricity = train_data['electricity'].copy(),
    gas = train_data['gas'].copy(),
  )
  
  df_train.to_csv(data_path_config['output']['train'])

  df_test = FeatureProcessor(
    data = test_data['test'].copy(),
    client = test_data['client'].copy(),
    historical_weather = test_data['historical_weather'].copy(),
    forecast_weather = test_data['forecast_weather'].copy(),
    electricity = test_data['electricity'].copy(),
    gas = test_data['gas'].copy(),
  )
  
  df_test.to_csv(data_path_config['output']['test'])
  