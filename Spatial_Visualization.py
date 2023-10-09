# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 13:50:43 2023

@author: zhangwenxiu
"""
import pandas as pd
import numpy as np
import os

#定义两个函数用于加载九段线和中国
import cartopy.io.shapereader as shpreader
def add_china(ax, **kwargs):
    proj = ccrs.PlateCarree()
    reader = shpreader.Reader(r'G:\mask\南海诸岛\nanhai.dbf')
    provinces = reader.geometries()
    ax.add_geometries(provinces, proj, **kwargs)
    reader.close()
    
def add_dashline(ax, **kwargs):
    proj = ccrs.PlateCarree()
    reader = shpreader.Reader(r"G:\mask\南海诸岛\nanhai.dbf")
    provinces = reader.geometries()
    ax.add_geometries(provinces, proj, **kwargs)
    reader.close()
    
    
import geopandas as gpd 
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.colors import ListedColormap, BoundaryNorm  
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
from matplotlib.ticker import AutoMinorLocator,MultipleLocator,FuncFormatter,LinearLocator,NullLocator,FixedLocator,IndexLocator,AutoLocator

data1=pd.read_csv('D:\\Desktop\\space.csv')
data1=data1[['lat','lon','O3']]
bounds =list(range(6,16,1))
colors = ['#000080', '#0000BC', '#0000F8','#0034FF', '#0070FF', '#00ACFF','#00E8FF', '#26FFDA', '#62FF9E', '#9EFF62', '#DAFF26', '#FFE800', '#FFAC00','#FF7000', '#FF3400', '#F80000','#BC0000']

cmap = ListedColormap(colors)
norms = BoundaryNorm(bounds, cmap.N)
projn = ccrs.LambertConformal(central_longitude=105, 
                              central_latitude=40,
                              standard_parallels=(25.0, 47.0))

fig = plt.figure(figsize=(4,3.5),dpi=120,facecolor="w")
ax = fig.add_subplot(projection=projn)
#ax.add_feature(cfeature.LAND.with_scale('50m'))####添加陆地######
#ax.add_feature(cfeature.COASTLINE.with_scale('50m'),lw=0.25)#####添加海岸线#########
#ax.add_feature(cfeature.RIVERS.with_scale('50m'),lw=0.25)#####添加河流######
#ax.add_feature(cfeature.LAKES.with_scale('50m'))######添加湖泊#####
#ax.add_feature(cfeature.BORDERS, linestyle='-',lw=0.25)####不推荐，我国丢失了藏南、台湾等领土############
#ax.add_feature(cfeature.OCEAN.with_scale('50m'))######添加海洋########
#ax.add_feature(cfeature.LAND, facecolor='white')
#ax.add_feature(cfeature.OCEAN)
#ax.add_feature(cfeature.LAKES.with_scale('110m'), facecolor='#BEE8FF')
ax.set_extent([80, 127, 17, 54], crs=ccrs.PlateCarree())
#long = np.linspace(72, 136, 128); lat = np.linspace(18, 54, 72)    
ax.spines['geo'].set_linewidth(.5)    
   
#添加geopandas 读取的地理文件
#add_dashline(ax, ec="black", linewidth=.2)
add_china(ax, ec="black", fc="None", linewidth=.2)
gls = ax.gridlines(draw_labels=True,dms = False,crs=ccrs.PlateCarree(), 
                   color='Gray', linestyle='dashed', linewidth=0.2, 
                   alpha=0.6,
                   y_inline=False, x_inline=False,
                   rotate_labels=False,xpadding=5,
                   xlabel_style={"size":14,'color':'w'},
                   ylabel_style={"size":14,'color':'w','rotation':90},
                   xlocs=range(80,140,10), ylocs=range(20,55,10)
                  )    

im=ax.scatter(data1["lon"],
               data1["lat"],
               s=6,c=data1["O3"],
               cmap=cmap,norm=norms,
               vmin=6,vmax =16,
               transform=ccrs.PlateCarree())
#cbar = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.1, orientation='horizontal/vertical',extend = 'both/max/min')
cbar = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.1, orientation='horizontal')
#显示间隔
#cbar.ax.yaxis.set_major_locator(AutoLocator(6))
#cbar.ax.yaxis.set_minor_locator(MultipleLocator(1))
cbar.ax.tick_params(labelsize=10)

gls.top_labels= False                      
gls.right_labels=False
ax2 = fig.add_axes([0.856, 0.042, 0.1, 0.25], projection = projn)
ax2.set_extent([105,125,2,23])
ax2.spines['geo'].set_linewidth(.2)    
    
#设置网格点
add_dashline(ax2, ec="black",  fc="None", linewidth=.2)
#add_china(ax2, ec="black", fc="None", linewidth=.2)
#ax2.add_feature(cfeature.LAND, facecolor='w')
plt.tight_layout()
plt.show() 
plt.savefig('plot\\aa.jpg', dpi=400,bbox_inches='tight',pad_inches =0.1)  #保存成jpg格式      
 
 

 
