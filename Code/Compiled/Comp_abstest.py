import os
import numpy as np
import colorsys
from osgeo import gdal, osr
from PIL import Image
import gc

def openGeo(pointer):
    """
    reads in a geotiff as image and stores coordinates in memory
        Input:
            Pointer: string (Pointer to file location)
        Output:
            img: Image type of PIL
            img.size: Tuple (row, col)
            geotrans: Tuple (GeoTransformations from GDAL)
            geoproj: Tuple (Projection Parameters from GDAL)
    """
    img = Image.open(pointer)
    srstif = gdal.Open(pointer)
    geotrans = srstif.GetGeoTransform()
    geoproj = srstif.GetProjection()
    return img, img.size, geotrans, geoproj

def RGBtoVal(img, dim):
    """
    Changes colourspace of a 3 band RGB image to a one band Value Image
        Input:
            img: Image type of PIL
            dim: Tuple (row, col of the PIL image)
        Output:
            ndarray: Representation of the image in numpy
    """
    if isinstance(img,Image.Image):
        r,g,b = img.split() #split bands
        Vdat = [] 
        for rd,gn,bl in zip(r.getdata(),g.getdata(),b.getdata()) :
            #v = colorsys.rgb_to_hsv(rd/255.,gn/255.,bl/255.)[2] #RGB to HSV (normalised)
            #Vdat.append(v)
            #h = colorsys.rgb_to_hsv(rd/255.,gn/255.,bl/255.)[0] #RGB to HSV (normalised)
            #Vdat.append(h)
            s = colorsys.rgb_to_hsv(rd/255.,gn/255.,bl/255.)[1] #RGB to HSV (normalised)
            Vdat.append(s)
        return np.reshape(np.array(Vdat),(dim[1],dim[0])) #return value as numpy array
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

def saveGeo(outName, dim, transform, project, inpdata):
    """
    Save np array as Geotiff (only for 1D Geotiff)
        Input:
            outName: Image type of PIL
            dim: Tuple (row, col of the PIL image)
            geotrans: Tuple (GeoTransformations from GDAL)
            geoproj: Tuple (Projection Parameters from GDAL)
            inpdata: ndarray (Representation of the image in numpy)
        Output:
            Nothing
    """
    driver = gdal.GetDriverByName("GTiff")
    saveTif = driver.Create(outName, dim[0], dim[1], 1, gdal.GDT_Float32)
    print("Dimension for out:", dim)
    saveTif.SetGeoTransform(transform)
    saveTif.SetProjection(project)
    saveTif.GetRasterBand(1).WriteArray(inpdata)
    saveTif.FlushCache()

def hist_match(after, before):
    """
        Normalisation of images based on histogram matching to the before image.
    Input:
        after: np.ndarray (image after disaster)
        before: np.ndarray (image before disaster)
    Returns:
        matched: np.ndarray
    """

    imgsize = after.shape #retrieve array size
    flata = after.ravel() #flatten input array
    flatb = before.ravel() #flatten reference array

    # get the set of unique pixel values
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

"""
for hide in hide:
    op = openGeo("highres.tif")
    op1 = openGeo("lowres.tif")
    test = RGBtoVal(op[0].resize(op1[1], resample=Image.LANCZOS), op1[1])
    test1 = RGBtoVal(op1[0], op1[1])
    histmat = hist_match(test,test1)
    subt = np.absolute(np.subtract(histmat, test1))
    np.savetxt("testabs.txt", subt, delimiter=',')
    saveGeo("testabs.tif",op1[1],op1[2],op1[3],subt)

    before = openGeo("RawTest3.tif")
    after = openGeo("RawTest4.tif")
    beforearr = np.asarray(before[0].getdata()).reshape(before[1][1],before[1][0])
    afterarr = np.asarray(after[0].getdata()).reshape(before[1][1],before[1][0])
    histmat = hist_match(afterarr, beforearr)
    print(histmat.size)
    print(before[2])
    print(before[3])
    subt = np.absolute(np.subtract(histmat, beforearr))
    print(subt.shape)
    saveGeo("testhistmatchtest.tif",before[1],before[2],before[3],subt)

listName = os.listdir("data/Optical/sat/before")
for name in listName:
    print(name)
    opb = openGeo("data/Optical/sat/before/" + name)
    opa = openGeo("data/Optical/sat/after/" + name)
    print("loaded")
    resamp = opa[0].resize(opb[1], resample=Image.LANCZOS)
    print("resized")
    valb = RGBtoVal(opb[0], opb[1])
    vala = RGBtoVal(resamp, opb[1])
    print("Value from RGB")
    histmat = hist_match(vala,valb)
    print("Hist matched")
    subt = np.absolute(np.subtract(histmat, valb))
    print("subtracted")
    saveGeo("data/Optical/results/S_sat_histmat_" + name,opb[1],opb[2],opb[3],subt)

op = openGeo("data/Optical/after/GeoRe/RescUAV_21017_BillyFolly_modified.tif")
op1 = openGeo("data/Optical/before/RescUAV_21017_BillyFolly.tif")
test = RGBtoVal(op[0], op[1])
test1 = RGBtoVal(op1[0], op1[1])
histmat = hist_match(test,test1)
subt = np.absolute(np.subtract(histmat, test1))
saveGeo("data/Optical/Hist_BillyFolly_Geo.tif",op1[1],op1[2],op1[3],subt)
"""

opb = openGeo("data/Optical/before/RescUAV_17917_Middle.tif")
opa = openGeo("data/Optical/after/GeoRe/RescUAV_17917_Middle_modified4.tif")
print("Loaded")
resamp = opa[0].resize(opb[1], resample=Image.LANCZOS)
print("Resampled")
print(resamp.size)
valb = RGBtoVal(opb[0], opb[1])
vala = RGBtoVal(resamp, opb[1])
print("Colourspace changed")
histmat = hist_match(vala,valb)
print("Histogram Matched")
subt = np.absolute(np.subtract(histmat, valb))
print("Subtracted")
saveGeo("data/Optical/S_Hist_Middle_.tif",opb[1],opb[2],opb[3],subt) 
