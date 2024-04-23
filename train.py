

# importing necessary libraries
import requests
import os
import datetime as dt
import csv
import shutil
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet.serialize import model_to_json, model_from_json
import math
import time
import pandas as pd
from prophet import *
import warnings
warnings.filterwarnings("ignore")

def cord(city):
# Initialize latitude and longitude
   dir = {
    'bangalore':(12.9767936 ,77.590082),
    'delhi':(28.6517178 ,77.2219388),
    'lucknow':(26.8381 ,80.9346001),
    'chennai':(13.0836939 ,80.270186)
    }
   return dir[city]

def get_perf(m, train, horizon=365):
    fcst_df = m.make_future_dataframe(periods=horizon)
    fcst = m.predict(fcst_df)


    perf_df = fcst[:-horizon].dropna(subset=['yhat'])
    train = train.dropna(subset=['y'])


    perf_df = perf_df.merge(train[['ds', 'y']], on='ds', how='inner')

    score_mae = mean_absolute_error(perf_df['y'], perf_df['yhat'])
    score_rmse = math.sqrt(mean_squared_error(perf_df['y'], perf_df['yhat']))

    print(score_mae)
    print(score_rmse)

    return score_rmse

#-----Weekly Dataset updation--------------
def weekly_update(city,model):
    lat, lon = cord(city)
    retrain_log_path="retrain/{}/{}_retrain_log.csv".format(model,city)
    df = pd.read_csv(retrain_log_path)
    # new_value1 = int(time.time())
    new_value1 =  1711737010
    print(new_value1)
    start = df['last updated date'].iloc[-1] + 60 * 60
    end=df['last updated date'].iloc[-1]
    print(df['last updated date'].iloc[-1]+ 7*24*60*60)
    while(new_value1>end):    #Updating will continue until it reaches the current time.
      end=end+ 7*24*60*60
    end=end- 7*24*60*60
    new_value2 = end
    df = df.append({'last updated date': new_value2 , 'retrain datetime' :new_value1}, ignore_index=True)
    df.to_csv(retrain_log_path, index=False)



    # requesting AQI data from API
    if model=='AQI':
          key='cdde2404b1a47c8b46ba44aae32246c6'
          url = f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat={lat}&lon={lon}&type=hour&start={start}&end={end}&appid={key}"
          response = requests.get(url)
          w = pd.DataFrame()
          if(response.status_code==200):
            data = response.json()
            for i in data['list']:
                # Extract the values from each dictionary and append to the DataFrame
                w = w.append({'aqi': i['main']['aqi'] ,
                'CO' :i['components']['co'],
                  'no2' :i['components']['no2'],
                  'o3' :i['components']['o3'],
                    'so2' :i['components']['so2'],
                    'pm2_5' :i['components']['pm2_5'],
                      'nh3' :i['components']['nh3'],
                      'dt' :dt.datetime.fromtimestamp(i['dt'])},
                        ignore_index=True)
                # print(city,model)
          else:
             print("sleep")
             time.sleep(5)

          data_path='versioning/weektwo/aqi/{}_aqi_csv.csv'.format(city)
          df_new = pd.read_csv(data_path)
          k=pd.concat([df_new, w])

    # requesting weather data from API
    else:
          key='LXJN9E7CP2T8RJP3Z5FFQRK5K'
          w = pd.DataFrame()
          while(end>=start):
              url=f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{lat},{lon}/{start}/{end}?key={key}&include=days&unitGroup=metric"
              response = requests.get(url)
              if(response.status_code==200):
                data = response.json()
                print("data",data )
                for d in data['days']:
                    # Extract the values from each dictionary and append to the DataFrame
                    w = w.append({
                        'datetime': dt.datetime.strptime(d['datetime'], '%Y-%m-%d'),
                        'tempmax': d['tempmax'],
                        'humidity': d['humidity'],
                        'windspeed': d['windspeed'],
                        'cloudcover': d['clouds'],
                        'conditions': d['weather'][0]['description'],
                    }, ignore_index=True)
                start=start+60*60
                # print(city,model)
              else:
                 print("sleep")
                 time.sleep(5)
                 start=start+60*60
          data_path='versioning/weektwo/heatwave/{}_temp_csv.csv'.format(city)
          df_new = pd.read_csv(data_path)
          df_new['datetime'] =  pd.to_datetime(df_new['datetime'])
          k=pd.concat([df_new, w])
    return k


















def prophet_AQI(city):

  one_data_file_name="versioning/weekone/aqi/{}_aqi_csv.csv".format(city)
  one_prediction_file_name="versioning/weekone/aqi/{}_aqi_csv_forecast.csv".format(city)
  one_prediction_model_name="versioning/weekone/aqi/{}_aqi_csv.json".format(city)
  


  two_data_file_name ="versioning/weektwo/aqi/{}_aqi_csv.csv".format(city)
  two_prediction_file_name="versioning/weektwo/aqi/{}_aqi_csv_forecast.csv".format(city)
  two_prediction_model_name="versioning/weektwo/aqi/{}_aqi_csv.json".format(city)

  three_data_file_name="versioning/weekthree/aqi/{}_aqi_csv.csv".format(city)
  three_prediction_file_name="versioning/weekthree/aqi/{}_aqi_csv_forecast.csv".format(city)
  three_prediction_model_name="versioning/weekthree/aqi/{}_aqi_csv.json".format(city)

  four_data_file_name="versioning/weekfour/aqi/{}_aqi_csv.csv".format(city)
  four_prediction_file_name="versioning/weekfour/aqi/{}_aqi_csv_forecast.csv".format(city)
  four_prediction_model_name ="versioning/weekfour/aqi/{}_aqi_csv.json".format(city)

  winner_prediction_model_name="winner/aqi/{}_aqi_csv.json".format(city)
  winner_prediction_file_name="winner/aqi/{}_aqi_csv_forecast.csv".format(city)

  

  #preprocessing     
  os.rename(three_prediction_model_name, four_prediction_model_name)
  os.rename(three_prediction_file_name, four_prediction_file_name)
  os.rename(three_data_file_name, four_data_file_name)

  os.rename(two_prediction_model_name, three_prediction_model_name)
  os.rename(two_prediction_file_name, three_prediction_file_name)
  os.rename(two_data_file_name, three_data_file_name)

  os.rename(one_prediction_model_name, two_prediction_model_name)
  os.rename(one_prediction_file_name, two_prediction_file_name)
  os.rename(one_data_file_name, two_data_file_name)

  #weekly update via api 
  df = weekly_update(city,'AQI')
  df.to_csv(one_prediction_file_name, index=False)



  df['time'] = pd.to_datetime(df['dt'], dayfirst=True)
  df['aqi'] = df['aqi'].interpolate(method='linear')
  data_prophet =df[['time', 'aqi']].rename(columns={'time': 'ds', 'aqi': 'y'})



  model_optimized = Prophet(
      changepoint_prior_scale=0.01,
      seasonality_prior_scale=1.0,
  )
  model_optimized.fit(data_prophet)

  rmse=get_perf(model_optimized,data_prophet)
  print(rmse)



  future = model_optimized.make_future_dataframe(periods=365) # horizon
  future['yearly'] = future['ds'].apply(lambda x: x.year - 1) # addition
  forecast = model_optimized.predict(future)
  with open(one_prediction_model_name, 'w') as fout:
    fout.write(model_to_json(model_optimized))  # Save model
  print("model saved")
  forecast.to_csv(one_prediction_file_name, index=False)
  print("prediction file saved")

  # #updating log
  df_log = pd.read_csv('content/aqi/log.csv')
  df_log.loc[3, city] = ''
  df_log[city] = df_log[city].shift(1)
  df_log.loc[0, city] = rmse
  df_log.to_csv('content/aqi/log.csv', index=False)

  #comparing value in log
  with open('content/aqi/log.csv', mode='r') as file:
      reader = csv.reader(file)
      header = next(reader)
      col_index = header.index(city)
      min = float('inf')
      min_row = None
      for i, row in enumerate(reader):
          value = float(row[col_index])
          if value < min:
              min = value
              min_row = i + 1  

  # #winner model
  if min_row==1:
    shutil.copy(one_prediction_model_name, winner_prediction_model_name)
    shutil.copy(one_prediction_file_name, winner_prediction_file_name) 
  elif min_row==2:
    shutil.copy(two_prediction_model_name, winner_prediction_model_name)
    shutil.copy(two_prediction_file_name, winner_prediction_file_name)
  elif min_row==3:
    shutil.copy(three_prediction_model_name, winner_prediction_model_name)
    shutil.copy(three_prediction_file_name, winner_prediction_file_name)
  else:
    shutil.copy(four_prediction_model_name, winner_prediction_model_name)
    shutil.copy(four_prediction_file_name, winner_prediction_file_name)  
  



# prophet_AQI('bangalore')
# prophet_AQI('chennai')
# prophet_AQI('delhi')
# prophet_AQI('lucknow')





def prophet_temp(city):
  one_prediction_model_name="versioning/weekone/heatwave/{}_temp_csv.json".format(city)
  one_prediction_file_name="versioning/weekone/heatwave/{}_temp_csv_forecast.csv".format(city)
  one_data_file_name="versioning/weekone/heatwave/{}_temp_csv.csv".format(city)
  


  two_prediction_model_name="versioning/weektwo/heatwave/{}_temp_csv.json".format(city)
  two_prediction_file_name="versioning/weektwo/heatwave/{}_temp_csv_forecast.csv".format(city)
  two_data_file_name="versioning/weektwo/heatwave/{}_temp_csv.csv".format(city)

  three_prediction_model_name="versioning/weekthree/heatwave/{}_temp_csv.json".format(city)
  three_prediction_file_name="versioning/weekthree/heatwave/{}_temp_csv_forecast.csv".format(city)
  three_data_file_name="versioning/weekthree/heatwave/{}_temp_csv.csv".format(city)

  four_prediction_model_name="versioning/weekfour/heatwave/{}_temp_csv.json".format(city)
  four_prediction_file_name="versioning/weekfour/heatwave/{}_temp_csv_forecast.csv".format(city)
  four_data_file_name="versioning/weekfour/heatwave/{}_temp_csv.csv".format(city)

  winner_prediction_model_name="winner/heatwave/winner/{}_temp_csv.json".format(city)
  winner_prediction_file_name="winner/heatwave/winner/{}_temp_csv_forecast.csv".format(city)

  

  #preprocessing     
  os.rename(three_prediction_model_name, four_prediction_model_name)
  os.rename(three_prediction_file_name, four_prediction_file_name)
  os.rename(three_data_file_name, four_data_file_name)

  os.rename(two_prediction_model_name, three_prediction_model_name)
  os.rename(two_prediction_file_name, three_prediction_file_name)
  os.rename(two_data_file_name, three_data_file_name)

  os.rename(one_prediction_model_name, two_prediction_model_name)
  os.rename(one_prediction_file_name, two_prediction_file_name)
  os.rename(one_data_file_name, two_data_file_name)

  #weekly update via api
  df = weekly_update(city,'heatwave') #latest data
  df.to_csv(one_data_file_name, index=False)

  df['time'] = pd.to_datetime(  df['datetime'], dayfirst=True)
  df['tmax'] = df['tempmax'].interpolate(method='linear')
  data_prophet = df[['time', 'tmax']].rename(columns={'time': 'ds', 'tmax': 'y'})


  model_optimized = Prophet(
    changepoint_prior_scale = 0.5,
      seasonality_prior_scale = 0.1

  )
  model_optimized.fit(data_prophet)

  rmse=get_perf(model_optimized,   data_prophet)
  print(rmse)

  future = model_optimized.make_future_dataframe(periods=365) # horizon
  future['yearly'] = future['ds'].apply(lambda x: x.year - 1) # addition
  forecast = model_optimized.predict(future)



  with open(one_prediction_model_name, 'w') as fout:
    fout.write(model_to_json(model_optimized))  # Save model
  print("model saved")
  forecast.to_csv(one_prediction_file_name, index=False)
  print("prediction file saved")
  

  #updating log
  df_log = pd.read_csv('content/heatwave/log.csv')
  df_log.loc[3, city] = ''
  df_log[city] = df_log[city].shift(1)
  df_log.loc[0, city] = rmse
  df_log.to_csv('content/heatwave/log.csv', index=False)



  #comparing value in log
  with open('content/heatwave/log.csv', mode='r') as file:
      reader = csv.reader(file)
      header = next(reader)
      col_index = header.index(city)
      min = float('inf')
      min_row = None
      for i, row in enumerate(reader):
          value = float(row[col_index])
          if value < min:
              min = value
              min_row = i + 1  

  #winner model
  if min_row==1:
    shutil.copy(one_prediction_model_name, winner_prediction_model_name)
    shutil.copy(one_prediction_file_name, winner_prediction_file_name) 
  elif min_row==2:
    shutil.copy(two_prediction_model_name, winner_prediction_model_name)
    shutil.copy(two_prediction_file_name, winner_prediction_file_name)
  elif min_row==3:
    shutil.copy(three_prediction_model_name, winner_prediction_model_name)
    shutil.copy(three_prediction_file_name, winner_prediction_file_name)
  else:
    shutil.copy(four_prediction_model_name, winner_prediction_model_name)
    shutil.copy(four_prediction_file_name, winner_prediction_file_name)  
  


prophet_temp('bangalore')
# prophet_temp('chennai')
# prophet_temp('delhi')
# prophet_temp('lucknow')