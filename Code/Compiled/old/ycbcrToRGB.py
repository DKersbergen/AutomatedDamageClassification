import os
from osgeo import gdal, osr

listName = os.listdir("data/Optical/after")
listCheck = os.listdir("data/Optical/after/warped")
for name in listName:
    print("gdalwarp -co compress=lzw -co predictor=2 data/Optical/after/" + name + " data/Optical/after/warped/" + name)
    if name in listCheck:
        print("pass")
        pass
    else:
        print("go")
        os.system("gdalwarp -co compress=lzw -co predictor=2 data/Optical/after/" + name + " data/Optical/after/warped/" + name)