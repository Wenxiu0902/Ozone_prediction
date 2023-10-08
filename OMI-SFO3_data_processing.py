# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 13:56:12 2021
@author: Summer
"""
import os
import h5py
import numpy as np
import pandas as pd
from multiprocessing import Pool
import multiprocessing
import datetime as dt

rootdir ='\\OMI\\2008\\'

def UTC_Time(utc_s):
    t0=dt.datetime.strptime('1993-01-01','%Y-%m-%d') 
    utc_s = round(utc_s/3600,0)*3600 
    t2=t0+dt.timedelta(seconds=utc_s)
    otherStyleTime = t2.strftime("%Y.%m.%d.%H")
    return(otherStyleTime)

def array_with_nans(array):
    """ 
    Extracts the array and replaces fillvalues and missing values with Nans
    """
    MissingValue=-1.2676506e+30
    array[array==MissingValue]=np.nan
    return array

def readhe5_file(he5):
    
    ##打开h5文件
   #print (rootdir+he5)
    f = h5py.File(rootdir+he5,'r')   
   #print (f)
    LAT=f['HDFEOS/SWATHS/O3Profile/Geolocation Fields/Latitude']
    LON=f['HDFEOS/SWATHS/O3Profile/Geolocation Fields/Longitude']
    TIME = f['HDFEOS/SWATHS/O3Profile/Geolocation Fields/Time']
    PRESS=f['HDFEOS/SWATHS/O3Profile/Geolocation Fields/Pressure']
    OZONE=f['HDFEOS/SWATHS/O3Profile/Data Fields/O3']
    QUALITY=f['HDFEOS/SWATHS/O3Profile/Data Fields/ProcessingQualityFlags']


    ##fv=OZONE.attrs['_FillValue']
    mv=OZONE.attrs['MissingValue']
    LAT=LAT[:,:]
    LON=LON[:,:]
    TIME=TIME[:]
    QUALITY=QUALITY[:,:]
    Press_max=PRESS[:,:,18]
    Press_min=PRESS[:,:,17]
    Oz=OZONE[:,:,17]
    
    Oz[Oz==mv]=np.nan
    Press_min[Press_min==mv]=np.nan
    Press_max[Press_max==mv]=np.nan
    '''
    Oz=array_with_nans(Oz)
    Press_min=array_with_nans(press_min)
    Press_max=array_with_nans(press_max)
    '''

    ##时间的转换 
    time=pd.DataFrame(TIME)
   #print(time)
    time.columns=['seconds']
    time['utc']=''
    for n in range(len(time)):
        time.loc[n,'utc']=UTC_Time(time.loc[n,'seconds'])
    
    O3_shape=Oz.shape
    for i in range(O3_shape[1]):
        ##print(i)
        time[i]=time['utc']
    time=time.drop(['seconds', 'utc'], axis=1)

    lat=pd.DataFrame(LAT)
    lon=pd.DataFrame(LON)
    ozone=pd.DataFrame(Oz)
    quality=pd.DataFrame(QUALITY)
    press_min=pd.DataFrame(Press_min)
    press_max=pd.DataFrame(Press_max)
    lat1=pd.melt(lat)
    lon1=pd.melt(lon)
    time1=pd.melt(time)
    quality1=pd.melt(quality)
    ozone1=pd.melt(ozone)
    press_min1=pd.melt(press_min)
    press_max1=pd.melt(press_max)
    
    Ozone=pd.concat([quality1.value,lon1.value,lat1.value,press_min1.value,press_max1.value,time1.value,ozone1.value],axis=1)
    Ozone.columns=['quality','lon','lat','press_min','press_max','time','O3']
    O3=Ozone[Ozone['quality']==0]
    O3=O3.drop(['quality'], axis=1)

    '''
    计算ppb（DU单位变ppb）    
    ppm=<vmr>i = 1.2672 Ni / DPi
    with Ni the layer-column in DU, DPi the pressure difference between the top and bottom of the
    layer in hPa and <vmr>I the average volume mixing ratio in ppmv.
    '''    
    O3['O3_ppb']=1.2672*O3['O3']*1000/(O3['press_max']-O3['press_min'])
    O3=O3.dropna(axis=0,how='any')
    O3.reset_index(inplace=True,drop=True)
    ##print(O3.loc[1,'time'])
    return(O3)


'''计算不同分辨率的栅格行列数'''
def cal_row_col(lon_0,lat_0,resolution,series_lon,series_lat):
    ##eg:col=((china_O31['lon'] - lon_0)/0.1).astype("int"))
    col=((series_lon - lon_0)/resolution).astype("int")   
    row=((series_lat - lat_0)/(-1.*resolution)).astype("int")   
    return (col,row)
  
Ozone_ppb= pd.DataFrame()
def write_file(file_list,num,year): 
    pool = multiprocessing.Pool()
    Ozone_ppb_list = pool.map_async(readhe5_file,file_list).get()
    Ozone_ppb = pd.concat(Ozone_ppb_list,axis=0,ignore_index=True)
    ##提取出中国区域
    china_O3= Ozone_ppb[(Ozone_ppb.lon<135)&(Ozone_ppb.lon>71.5)&(Ozone_ppb.lat<52.5)&(Ozone_ppb.lat>3)]
    china_O3.reset_index(inplace=True,drop=True)
    china_O31=china_O3[['lon','lat','time','O3_ppb']]

    col1,row1=cal_row_col(73.5,53.5,0.1,china_O31['lon'],china_O31['lat'])
    china_O31.insert(4,'col',col1)
    china_O31.insert(5,'row',row1)    
    china_O31['row'] = china_O31['row'].astype("int")
    china_O31['col'] = china_O31['col'].astype("int")   
    china_O31.to_csv('OMI/Processed/China/'+str(year)+'/'+str(num)+'.csv',header=0)
    pool.close()
    pool.join()
   
allfile_list = os.listdir(rootdir) #列出文件夹下所有的目录与文件
box_number = 300
allfile_number = len(allfile_list)
allfile_boxnumber = int(allfile_number / box_number ) + 1

if __name__ == "__main__": 
        
      for i in range(allfile_boxnumber):
        temp_list = allfile_list[box_number*i:box_number*(i+1)]
        if temp_list != []:
            write_file(temp_list,i,2008)

