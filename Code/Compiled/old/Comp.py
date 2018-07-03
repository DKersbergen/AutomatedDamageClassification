import os
import numpy as np
import colorsys
from osgeo import gdal, osr
from PIL import Image

def openGeo(pointer):
    """
    reads in a geotiff as image and stores coordinates in memory
    """
    img = Image.open(pointer)
    srstif = gdal.Open(pointer)
    geotrans = srstif.GetGeoTransform()
    geoproj = srstif.GetProjection()
    return img, img.size, geotrans, geoproj

def RGBtoVal(img, dim):
    """
    Changes colourspace of a 3 band RGB image to a one band Value Image
    """
    if isinstance(img,Image.Image):
        r,g,b = img.split()
        Vdat = [] 
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
    """
    Save np array as Geotiff
    """
    driver = gdal.GetDriverByName("GTiff")
    saveTif = driver.Create(outName, row, col, bands, gdal.GDT_Int16)
    saveTif.SetGeoTransform(transform)
    saveTif.SetProjection(project)
    saveTif.GetRasterBand(1).WriteArray(inpdata)
    saveTif.FlushCache()

def hist_match(after, before):
    """
        Normalisation of images based on histogram matching to the before image.
    Input:
    -----------
        after: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        before: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed after image
    """

    imgsize = after.shape #retrieve array size
    flata = after.ravel() #flatten input array
    flatb = before.ravel() #flatten reference array

    # get the set of unique pixel values and their corresponding indices and
    # counts
    bin_idx, a_counts = np.unique(flata, return_inverse=True, return_counts=True)[1:]
    b_values, b_counts = np.unique(flatb, return_counts=True)

    # take the cumulative sum of the counts and normalise by the number of pixels
    # to get the empirical CDF for the after and before images
    a_quantiles = np.cumsum(a_counts).astype(np.float64)
    a_quantiles /= a_quantiles[-1]
    b_quantiles = np.cumsum(b_counts).astype(np.float64)
    b_quantiles /= b_quantiles[-1]

    # linear interpolation of pixel values in the before image
    # to correspond most closely to the quantiles in the source image
    interp_b_values = np.interp(a_quantiles, b_quantiles, b_values)

    return interp_b_values[bin_idx].reshape(imgsize)

print(np.absolute(np.array([[1,1,1],[-5,-60,-10]])))

"""
op = openGeo("highres.tif")
op1 = openGeo("lowres.tif")
test = RGBtoVal(op[0].resize(op1[1], resample=Image.LANCZOS), op1[1])
test1 = RGBtoVal(op1[0], op1[1])
histmat = hist_match(test,test1)
subt = np.subtract(histmat, test1)
print(subt)
np.savetxt("test.txt", subt, delimiter=',')

saveGeo("test1.tif",op1[1][0],op1[1][1],1,op1[2],op1[3],subt)
"""