# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 14:52:44 2021

@author: zhangwenxiu
"""
import os
import numpy as np
import pandas as pd
import netCDF4 as nc
from netCDF4 import Dataset
import multiprocessing
from multiprocessing import Pool
import datetime as dt

'''计算不同分辨率的栅格行列数'''
def cal_row_col(lon_0,lat_0,resolution,series_lon,series_lat):
    ##eg:col=((china_O31['lon'] - lon_0)/0.1).astype("int"))
    col=((series_lon - lon_0)/resolution).astype("int")   
    row=((series_lat - lat_0)/(-1.*resolution)).astype("int")   
    return (col,row)
    
'''读取MEE站点信息'''
station_file = 'MEE_ONLYO3_original_observation_data\station_mee_china_0.1.csv'
df_station= pd.read_csv(station_file,header=0,sep=',') 
df_station=df_station[['station','col','row']]


'''开始读写数据'''
rootdir ='Y:\\origin\\ERA5LAND\\50variables_hourly\\2021-\\'
allfile_list = os.listdir(rootdir)
nc_obj=Dataset('land.20210102.nc')

##print(nc_obj.variables.keys())
##print(nc_obj.variables['ssrd'])  
Lon=nc_obj.variables['longitude'][:].data          
Lat=nc_obj.variables['latitude'][:].data
col1,row1=cal_row_col(70,55,0.1,Lon,Lat)
        
#经纬度变换
LON2,LAT2=np.meshgrid(Lon,Lat)
COL2,ROW2=np.meshgrid(col1,row1)
LON= np.empty([24,701,701])
LAT= np.empty([24,701,701])
COL= np.empty([24,701,701])
ROW= np.empty([24,701,701])
for i in range(24):
    LON[i]=LON2
    LAT[i]=LAT2 
    COL[i]=COL2
    ROW[i]=ROW2

LON=pd.DataFrame(LON.reshape(-1,1))
LAT=pd.DataFrame(LAT.reshape(-1,1))
COL=pd.DataFrame(COL.reshape(-1,1))
ROW=pd.DataFrame(ROW.reshape(-1,1))

missvalue=nc_obj.variables['e'].missing_value



def read_nc_file(nc_file,n):
    
    nc_obj_file=rootdir+nc_file
    nc_obj=Dataset(nc_obj_file)
    #时间数据维度的转换
    real_time=nc.num2date(nc_obj.variables['time'],units=nc_obj['time'].units).data
    TIME= np.empty([24,701,701], dtype = object)
    for i in range(24):TIME[i,:,:]=real_time[i]
    TIME3=pd.DataFrame(TIME.reshape(-1,1)) 
    #print(LON.shape,LAT.shape,COL.shape,ROW.shape,TIME3.shape)
    #气象变量
    pd_ssrd=pd.DataFrame(nc_obj.variables['ssrd'][:,:].data.reshape(-1,1))
    pd_Evap=pd.DataFrame(nc_obj.variables['e'][:,:].data.reshape(-1,1))
    pd_tp=pd.DataFrame(nc_obj.variables['tp'][:,:].data.reshape(-1,1))
    pd_t2m=pd.DataFrame(nc_obj.variables['t2m'][:,:].data.reshape(-1,1))
    pd_u10=pd.DataFrame(nc_obj.variables['u10'][:,:].data.reshape(-1,1))
    pd_v10=pd.DataFrame(nc_obj.variables['v10'][:,:].data.reshape(-1,1))
    pd_d2m=pd.DataFrame(nc_obj.variables['d2m'][:,:].data.reshape(-1,1))
    pd_sp=pd.DataFrame(nc_obj.variables['sp'][:,:].data.reshape(-1,1))
    
    pd_ssrd[pd_ssrd==missvalue]=np.nan
    pd_Evap[pd_Evap==missvalue]=np.nan
    pd_tp[pd_tp==missvalue]=np.nan
    pd_t2m[pd_t2m==missvalue]=np.nan
    pd_u10[pd_u10==missvalue]=np.nan
    pd_v10[pd_v10==missvalue]=np.nan
    pd_d2m[pd_d2m==missvalue]=np.nan
    pd_sp[pd_sp==missvalue]=np.nan
    

    pd_all=pd.concat([LAT,LON,COL,ROW,TIME3,pd_ssrd,pd_Evap,pd_tp,pd_t2m,pd_u10,pd_v10,pd_d2m,pd_sp],axis=1)
    pd_all.columns=['lat','lon','col','row','time','ssrd','eva','tp','t2m','u10','v10','d2m','sp']
    
    ##pd_china= pd_all[(pd_all.lon<140)&(pd_all.lon>70)&(pd_all.lat<55)&(pd_all.lat>0)]
    ##missvalue=nc_obj.variables['e'].missing_value
    ##pd_china=pd_china.dropna(axis=0,how='any')
    ##pd_china.reset_index(inplace=True,drop=True)
    ##df3=df_merge.groupby(['time','row','col'], as_index=False)['lat','lon','ssrd','eva','tp','t2m','u10','v10','d2m','sp'].mean()
    
    #只提取站点区域训练
    pd_train=pd.merge(pd_all,df_station,how='inner',on=['row','col'])
    pd_train=pd_train.dropna(axis=0,how='any')
    pd_train.reset_index(inplace=True,drop=True)
    pd_train1=pd_train.groupby(['time','row','col'])[['lat','lon','ssrd','eva','tp','t2m','u10','v10','d2m','sp']].mean()
    pd_train1.to_csv('\\2021\\'+ str(n) +'.csv')
    return


if __name__ == "__main__": 
    allfile_list = os.listdir(rootdir)
    n_list = list(range(len(allfile_list)))
    pool = multiprocessing.Pool(processes=20) 
    c=list(zip(allfile_list,n_list))
    pool.starmap(read_nc_file,c)
    pool.close()
    pool.join()


