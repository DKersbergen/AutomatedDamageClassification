
from osgeo import gdal, osr
import numpy as np
from skimage import color
from skimage.viewer import ImageViewer as ImgV

def array(name):
    """
    reads in a geotiff as ndarray representing an RGB image of shape (..,..,3)
    """
    geo = gdal.Open(name)
    arr = geo.ReadAsArray()
    shp = arr.shape
    print(shp)
    out = np.rot90(np.array((arr[0].ravel(),arr[1].ravel(),arr[2].ravel()), dtype=np.uint8))
    out1 = np.reshape(out, (shp[1],shp[2],shp[0]) )
    print(out1)

    return out1
    
def low_to_high(lowres, highres):
    """
    Resamples the low resolution optical image
    to match the dimensions of the higher resolution image
    """
    pass


def high_to_low(highres, lowres):
    """
    Resamples the high resolution optical image
    to match the dimensions of the lower resolution image
    This method is preffered as it does not create artifacts
    """
    pass

lowres = array("lowres.tif")
#highres = array("highres.tif")

lowreshsv = color.convert_colorspace(lowres,'RGB','HSV')
print(lowres)

ImgV(lowres).show()
print(lowreshsv)

ImgV(lowreshsv).show()
#print(np.transpose(lowres[1].ravel()))
#print(lowreshsv)
print('done')

