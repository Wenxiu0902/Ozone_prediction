# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 14:24:47 2022

@author: zhangwenxiu
"""
print('load module ...')
import numpy as np
from numpy import concatenate
from math import sqrt
import os
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import ReduceLROnPlateau
from scipy import stats
from pdb import set_trace
import glob
from scipy import stats
from scipy.stats import gaussian_kde

def rollroll(N, Nroll):
        aa = np.arange(N)
        temp = np.arange(N - Nroll + 1)
        A, B = np.meshgrid(aa, temp)
        T = (A - B)[:, -Nroll:]
        return T[::-1,:].flatten().copy()

#calculate max and min to Standardization

list_csvs = glob.glob('\\test_data\\*.csv')
csv_min=[]
csv_max=[]
for icsv in list_csvs[:]:   
    print('Processing ' + str(icsv[-7:]))
    pd_data=pd.read_csv(icsv, header=0)
    data_max=pd_data.max() #col_max
    data_min=pd_data.min() #col_min
    csv_min.append(data_min)
    csv_max.append(data_max)
csv_min= pd.concat(csv_min, axis=1)
csv_max= pd.concat(csv_max, axis=1)

Max_features=csv_max.max(axis=1)[['row', 'col', 'gdp', 'pop', 'grass',
    'forest', 'urban', 'crop', 'ssrd', 'tp', 't2m', 'u10', 'v10', 'd2m',
    'sp', 'DOY', 'to3', 'omi']]
Min_features=csv_min.min(axis=1)[['row', 'col', 'gdp', 'pop', 'grass',
    'forest', 'urban', 'crop', 'ssrd', 'tp', 't2m', 'u10', 'v10', 'd2m',
    'sp', 'DOY', 'to3', 'omi']]
Max_label=csv_max.max(axis=1)[['MEE']].values
Min_label=csv_min.min(axis=1)[['MEE']].values

Max_features= pd.Series([335, 575, 583973, 44643, 1.0, 1.0, 1.0, 1.0,32739839,  0.244536, 315,13, 20, 303,101218, 213, 389.3,  38.78], index=['row', 'col', 'gdp', 'pop', 'grass', 'forest', 'urban', 'crop', 'ssrd', 'tp', 't2m', 'u10', 'v10', 'd2m', 'sp', 'DOY', 'to3', 'sfo3'])
Min_features=pd.Series([31, 24, 0.0, 0.38, 0.0, 0.0, 0.0, 0.0, 494.4, 0, 271.95, 0.000012,  0.000026,   259.8035,  56998.07, 183, 243.23, 38.77],index=['row', 'col', 'gdp', 'pop', 'grass', 'forest', 'urban', 'crop', 'ssrd', 'tp', 't2m', 'u10', 'v10', 'd2m', 'sp', 'DOY', 'to3', 'sfo3'])
Max_label=150
Min_label=0.5

feature_range=[0,1]
def make_3D_data(csv_dir, backTime):  
    print('Processing ' + str(csv_dir[-7:]))
    df_data=pd.read_csv(csv_dir, header=0)
    df_data=df_data[['utc_time', 'row','col', 'lat', 'lon', 'dem', 'gdp', 'pop',  'grass', 'forest',
           'urban', 'crop', 'ssrd', 'tp', 't2m', 'u10', 'v10', 'd2m', 'sp', 'DOY',
           'to3', 'omi', 'MEE']]
    df_data.columns=['utc_time', 'row', 'col','lat','lon', 'dem','gdp', 'pop', 'grass',
        'forest', 'urban', 'crop', 'ssrd', 'tp', 't2m', 'u10', 'v10', 'd2m',
        'sp', 'DOY', 'to3', 'sfo3','MEE']
    df_data.set_index(["utc_time"], inplace=True)
    df_features = df_data[['row', 'col', 'gdp', 'pop', 'grass',
        'forest', 'urban', 'crop', 'ssrd', 'tp', 't2m', 'u10', 'v10', 'd2m',
        'sp', 'DOY', 'to3', 'sfo3']]
    df_label = df_data['MEE']

    '''
    from matplotlib import pyplot
    values = df_data.values
    groups = list(range(15,22))
    i = 1
    pyplot.figure(figsize=(20,25))
    for group in groups:
        print(group)
        pyplot.subplot(len(groups), 1, i)
        pyplot.plot(values[:, group])
        pyplot.title(df_data.columns[group], y=0.7, loc='right')
        i += 1
    pyplot.show()
    '''
    #- Standardization for features, label will be scaled after
    feature_range=[0,1]
    arr_features_values = df_features.values
    x_std = (df_features-Min_features)/(Max_features-Min_features)
    arr_features_scaled_values = x_std*(feature_range[1]-feature_range[0]) + feature_range[0]
    #scaler_features = MinMaxScaler(feature_range=(0, 1))
    #arr_features_scaled_values = scaler_features.fit_transform(arr_features_values)
    
    df_scaled_features = pd.DataFrame(arr_features_scaled_values,columns = df_features.columns)
    df_scaled_features.set_index(df_data.index, inplace=True)
    #df_data_temp: features were scaled, label wasn't
    df_data_temp = pd.concat([df_label, df_scaled_features], axis=1) # (759,20)
    #print(df_data_temp.describe()) 

    #- make data for LSTM modelc
    #backTime = 24 # can be 2,3,4 ... 24,25, ...
    #- N = total sample, Nroll = backtime(hours)
    #- This function returns sequence like: 0,1,2,3, 1,2,3,4, 2,3,4,5
    
    #- sample: a=df[df_f].values[[0,1,2,3,1,2,3,4,2,3,4,5]].reshape(3,4,len(df_f))
    #- For data in one station
    roll_list = rollroll(len(df_data_temp), backTime)
    Nsample_orig = len(df_data_temp)
    Nfeature = len(df_scaled_features.keys())
    array_feature_rolled_3D = df_scaled_features.values[roll_list].reshape(Nsample_orig - backTime + 1, backTime, Nfeature)
      
    #- Standardization for label:   
    #array_label_rolled_not_scaled =df_data_temp.MEE.values[backTime-1:].reshape(-1, 1)   
    array_label_rolled_not_scaled =pd.DataFrame(df_data_temp.MEE.values[backTime-1:].reshape(-1, 1) )   
    #y_std = (array_label_rolled_not_scaled-Min_label)/(Max_label-Min_label)
    #arr_label_scaled_values = (y_std*(feature_range[1]-feature_range[0]) + feature_range[0]).values
        
    #scaler_label = MinMaxScaler(feature_range=(0, 1))
    #array_label_rolled = scaler_label.fit_transform(array_label_rolled_not_scaled.reshape(-1, 1))
    #- (757, 3, 19) (759, 1)
    
    return array_feature_rolled_3D, array_label_rolled_not_scaled

if __name__ == "__main__":
    
    #backtime=[1,2,6,12,24,36,48,72]
    #backtime=np.array(backtime)
    backTime = 24
    #model_name = './test/lstm_24_test.h5'
    #- Make data
    print('preparing data ...')

    train_list=glob.glob('train_data\\*.csv')
    test_list=glob.glob('test_data\\*.csv')

    list_train_x = []
    list_train_label = []
    list_test_x= []
    list_test_label = []
    
    
    #devide training and test set
    #num=int(len(list_csvs)*0.2)
    #for icsv in list_csvs[:-1]:
    for icsv in train_list[:]:
        train_x, train_label = make_3D_data(icsv, backTime)
        list_train_x.append(train_x)
        list_train_label.append(train_label)
    #set_trace()
    train_x = np.concatenate(list_train_x, axis=0).astype('float32')
    train_label = np.concatenate(list_train_label, axis=0).astype('float32')
    
    train_x_1=train_x.reshape(-1,train_x.shape[2])
    train_x_1=pd.DataFrame(train_x_1)
    train_label_1=pd.DataFrame(train_label)
    #train_x_1.to_csv('./processing_data/train_x_2.csv',index=0)
    #train_label_1.to_csv('./processing_data/train_label_2.csv',index=0)

    for icsv in test_list[:]:
        test_x, test_label= make_3D_data(icsv, backTime)
        list_test_x.append(test_x)
        list_test_label.append(test_label)

    test_x = np.concatenate(list_test_x, axis=0).astype('float32')
    test_label = np.concatenate(list_test_label, axis=0).astype('float32')
    
    test_x_1=test_x.reshape(-1,test_x.shape[2])
    test_x_1=pd.DataFrame(test_x_1)
    test_label_1=pd.DataFrame(test_label)
    #test_x_1.to_csv('./processing_data/test_x_2.csv',index=0)
    #test_label_1.to_csv('./processing_data/test_label_2.csv',index=0)
    #set_trace()

    nhid=50
    #- Build model
    model = Sequential()
    model.add(LSTM(nhid, input_shape=(train_x.shape[1], train_x.shape[2]), return_sequences = True))
    #model.add(LSTM(nhid, input_shape=(train_x.shape[1], nhid), return_sequences = True))
    model.add(LSTM(nhid, input_shape=(train_x.shape[1], nhid)))
    model.add(Dense(1)) # fully connected layer
    model.compile(loss='mse', optimizer='adam')

    
    #- train model
    reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, patience = 1, verbose=0, mode='auto', min_delta=0.00001, cooldown=0, min_lr=0)
    history = model.fit(train_x, train_label, epochs=50, batch_size=3000, validation_data=(test_x, test_label),callbacks=[reduce_lr], verbose=2, shuffle=True)

    #- plot loss curve
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()

    #- make prediction
    
    Predicted_label = model.predict(np.array(test_x))
    #Predicted_label_scaled=((Predicted_label-feature_range[0])/(feature_range[1]-feature_range[0]))*(Max_label-Min_label)+Min_label
    RMSE = sqrt(mean_squared_error(Predicted_label, test_label))


    test_label_1 = test_label.reshape(test_label.shape[0])
    Predicted_label_1 = Predicted_label.reshape(Predicted_label.shape[0])
    R2 = np.corrcoef(Predicted_label_1, test_label_1)[0][1] **2
    
    print('cv: %.2f' % n)
    print('Test RMSE: %.3f' % RMSE)
    print('r^2: %.3f' %(R2))
    
    
    R2_1.append(R2)
    RMSE_1.append(RMSE)
    
    #model.save(model_name)
      #load model:
      #aaa = keras.models.load_model('lstm.h5')
      
      #- make prediction
      #set_trace()
  
      #train_label = train_label.reshape(train_label.shape[0])
      #Predicted_label = Predicted_label.reshape(Predicted_label.shape[0])

      predict_pm = np.array(Predicted_label_1)
      label_pm = np.array(test_label_1)
  
      # scatter plot
      fig = plt.figure(figsize = (7.5, 6))
      ax = fig.subplots(nrows = 1, ncols = 1)
      y = label_pm
      x = predict_pm
      mse = mean_squared_error(x,y)
      rmse = np.sqrt(mse)
      test_r2_all = np.corrcoef(x, y)[0][1] **2
  
      # Calculate the point density
      xy = np.vstack([x,y])
      z = stats.gaussian_kde(xy)(xy)
      # Sort the points by density, so that the densest points are plotted last
      idx = z.argsort()
      x, y, z = x[idx], y[idx], z[idx]
      CS = ax.scatter(x, y,c=z, s=20, cmap='Spectral_r')
  
      # 1:1 line
      #ax.set_xlim(0,1)
      #ax.set_ylim(0,1)
      ystart, yend = ax.get_ylim()
      xstart, xend = ax.get_xlim()
      s11 = np.max([xstart, ystart])
      e11 = np.min([yend, yend])
      nparr11 = np.linspace(s11, e11, 100)
      fig.colorbar(CS)
      ax.plot(nparr11, nparr11,linestyle='--',linewidth=1.2, color = "grey", alpha = 0.8)
  
      #Fitting line (linear)
      parameter = np.polyfit(x, y, 1)
      y2 = parameter[0] * x + parameter[1]
      ax.plot(x, y2,linestyle='-',linewidth=1, color = "black")
  
      #方程拼接
      deg = 1
      aa=''
      for i in range(deg+1): 
          bb=round(parameter[i],2)
          if bb>0:
              if i==0:
                  bb=str(bb)
              else:
                  bb='+'+str(bb)
          else:
                  bb=str(bb)
          if deg==i:
                  aa=aa+bb
          else:
              aa='y =' + aa + ' ' + bb + 'x^' + str(deg-i)  
  
      ax.text(0.05, 0.9, aa, fontsize=12, verticalalignment = "bottom", horizontalalignment = "left", transform = ax.transAxes)
      ax.text(0.05, 0.85,  'R2 = ' + str(round(test_r2_all,2)),fontsize=12, verticalalignment = "bottom", horizontalalignment = "left", transform = ax.transAxes)
      
      ax.text(0.05, 0.8, 'Number = ' + str(len(x)),fontsize=12,  verticalalignment = "bottom", horizontalalignment = "left", transform = ax.transAxes)
      ax.text(0.05, 0.75, 'rmse = ' + str(round(rmse,2)), fontsize=12, verticalalignment = "bottom", horizontalalignment = "left", transform = ax.transAxes)
  
      fig.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95, wspace =0.05, hspace =0.05)
      #fig.text(0.04, 0.38, "O3 observations (ug/" +  "$\mathregular{m^{3}}$)" , rotation=90,fontsize=13)
      #fig.text(0.35, 0.05, "O3 predictions (ug/" +  "$\mathregular{m^{3}}$)", fontsize=13)
      fig.text(0.04, 0.38, "O3 observations " , rotation=90,fontsize=13)
      fig.text(0.35, 0.05, "O3 predictions ", fontsize=13)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
      #ax.legend()
      #plt.show()
      plt.savefig('M:\\zhangwenxiu\\test\\data\\plot\\plot_cv_'+str(n)+'.png')    
      
    



