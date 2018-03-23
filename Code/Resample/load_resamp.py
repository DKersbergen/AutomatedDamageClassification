import os
import numpy as np
import colorsys
from osgeo import gdal, osr
from PIL import Image

def openGeo(pointer):
    """"
    reads in a geotiff as image and stores coordinates in memory
    """
    img = Image.open(pointer)
    srstif = gdal.Open(pointer)
    geotrans = srstif.GetGeoTransform()
    geoproj = srstif.GetProjection()
    return img, img.size, geotrans, geoproj

def RGBtoVal(img, dim):
    print(dim)
    if isinstance(img,Image.Image):
        r,g,b = img.split()
        Vdat = [] 
        ValIm = Image.new("L",dim)
        for rd,gn,bl in zip(r.getdata(),g.getdata(),b.getdata()) :
            v = colorsys.rgb_to_hsv(rd/255.,gn/255.,bl/255.)[2]
            Vdat.append(int(v*255.))
        return np.reshape(np.array(Vdat),(dim[1],dim[0]))
        #ValIm.putdata(Vdat)
        #return ValIm
    else:
        raise TypeError("Expected img to be an instance of Image.Image")
        

def resample(low, lowdim, high, highdim, updo):
    if updo == "UP":
        return low.resize(highdim, resample=Image.LANCZOS)
    elif updo == "DOWN":
        return high.resize(lowdim, resample=Image.LANCZOS)
    else:
        raise ValueError("Expected 'UP' or 'DOWN'")
    return 

def saveGeo(outName, row, col, bands, transform, project, inpdata):
    driver = gdal.GetDriverByName("GTiff")
    saveTif = driver.Create(outName, row, col, bands, gdal.GDT_Byte)
    saveTif.SetGeoTransform(transform)
    saveTif.SetProjection(project)
    print(row,col,inpdata.shape)
    saveTif.GetRasterBand(1).WriteArray(inpdata)
    saveTif.FlushCache()
    
#lowres = Image.open("lowres.tif")
#highres = Image.open("highres.tif")

op = openGeo("highres.tif")
test = RGBtoVal(op[0], op[1])
saveGeo("test.tif",op[1][0],op[1][1],1,op[2],op[3],test)

"""" 
print(list(highres.getdata()))
vallow = RGBtoVal(lowres, lowres.size)
valhigh = RGBtoVal(highres, highres.size)
resamp = valhigh.resize(lowres.size, resample=Image.LANCZOS)
resampup = vallow.resize(highres.size, resample=Image.LANCZOS)
vallow.show()
valhigh.show()
resamp.show()
resampup.show()
"""
"""
hdwn = dwn(highres, (hshp[1]/lshp[1],hshp[2]/lshp[2],1))
print(hdwn.shape)
ImgV(hdwn).show()
#print(np.transpose(lowres[1].ravel()))
#print(lowreshsv)
print('done')

file = "path+filename"
ds = gdal.Open(file)
band = ds.GetRasterBand(1)
arr = band.ReadAsArray()
[cols, rows] = arr.shape
arr_out = numpy array
driver = gdal.GetDriverByName("GTiff")
outdata = driver.Create(outFileName, rows, cols, 1, gdal.GDT_UInt16)
outdata.SetGeoTransform(ds.GetGeoTransform())##sets same geotransform as input
outdata.SetProjection(ds.GetProjection())##sets same projection as input
outdata.GetRasterBand(1).WriteArray(arr_out)
outdata.GetRasterBand(1).SetNoDataValue(10000)##if you want these values transparent
outdata.FlushCache() ##saves to disk!!
outdata = None
band=None
ds=None
"""