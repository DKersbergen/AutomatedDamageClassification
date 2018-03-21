
from osgeo import gdal, osr
import numpy as np
from skimage import color
from skimage.transform import downscale_local_mean as dwn
from skimage.viewer import ImageViewer as ImgV
from PIL import Image

def array(name):
    """
    reads in a geotiff as ndarray representing an RGB image of shape (..,..,3)
    """
    geo = gdal.Open(name)
    arr = geo.ReadAsArray()
    shp = arr.shape
    out = np.rot90(np.array((arr[0].ravel(),arr[1].ravel(),arr[2].ravel()), dtype=np.uint8))
    out1 = np.reshape(out, (shp[1],shp[2],shp[0]))
    return out1,shp

def high_to_low(highres, lowres):
    """
    Resamples the high resolution optical image
    to match the dimensions of the lower resolution image
    This method is prefered over the reverse as it does not create artifacts
    """
    pass

lowres,lshp = array("lowres.tif")
highres,hshp = array("highres.tif")

#lowreshsv = color.convert_colorspace(lowres,'RGB','HSV')
ImgV(lowres).show()
ImgV(highres).show()
print(highres.shape, lshp)

im = Image.open("lowres.tif")
im.show()
print(im)

hdwn = dwn(highres, (hshp[1]/lshp[1],hshp[2]/lshp[2],1))
print(hdwn.shape)
ImgV(hdwn).show()
#print(np.transpose(lowres[1].ravel()))
#print(lowreshsv)
print('done')

