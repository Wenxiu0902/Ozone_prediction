#author: Zhangwenxiu
#date: 9/2 2020
#code for conversion OMI...he5 files to GTiff files

library(sp)
library(raster)
library(ncdf4)
library(rgdal)

# note the "/"
# change the OMI file path and outfile path
rm(list = ls()); gc() # 清空内存
OMIPath <- "E:/OMI/HE5-TO-GeoTiff/2019/HE5"  
outPath <- "E:/OMI/HE5-TO-GeoTiff/2019/CHINA_GEOTIFF/"
fileList <- list.files(OMIPath,pattern = ".he5")
for(n in 1:length(fileList)){
  print(fileList[n])
  OMIName <- paste(OMIPath,'/',fileList[n],sep = '')
  OMI <- raster(OMIName, ncdf = TRUE)
  tempOMI <- flip(OMI, direction = 'y')
  OMIMatrix <- as.matrix(tempOMI)
  OMI_r <- raster(OMIMatrix, xmn = 70, xmx = 140, ymn = 15, ymx = 55)
  # outname
  fileName <- substr(fileList[n],1,28)
  outName = paste(outPath, fileName, sep = '')
  writeRaster(OMI_r,outName,format = "GTiff")
}
  
