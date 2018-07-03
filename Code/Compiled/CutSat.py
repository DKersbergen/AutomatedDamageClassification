import os
from osgeo import gdal, ogr

def clipRasterPoly(rasterPath, polyPath, outPath):
    print("gdalwarp -q -cutline " + polyPath + " -crop_to_cutline " + " -of GTiff " + rasterPath + " " + outPath)
    os.system("gdalwarp -q -cutline " + polyPath + " -crop_to_cutline " + " -of GTiff " + rasterPath + " " + outPath)

def createClipPol(inPath, field):
    print(inPath)
    driverSHP = ogr.GetDriverByName("ESRI Shapefile")
    ds = driverSHP.Open(inPath)
    try:
        if ds != None:
            print("pass")
    except:
        print("layer is empty")
    layer = ds.GetLayer()

    sptlRef = layer.GetSpatialRef()

    for feature in layer:
        fieldVal = feature.GetField(field)
        outDs = driverSHP.CreateDataSource("data/Optical/Sat/clipFeatures/" + str(fieldVal) + ".shp")
        outLayer = outDs.CreateLayer("data/Optical/Sat/clipFeatures/" + str(fieldVal) + ".shp", srs=sptlRef, geom_type = ogr.wkbPolygon)
        outDfn = outLayer.GetLayerDefn()
        inGeom = feature.GetGeometryRef()
        outFeat = ogr.Feature(outDfn)
        outFeat.SetGeometry(inGeom)
        outLayer.CreateFeature(outFeat)

def clipRasters(inPathSHP, inPathTIF, field):
    driverSHP = driverSHP = ogr.GetDriverByName("ESRI Shapefile")
    ds = driverSHP.Open(inPathSHP)
    try:
        if ds != None:
            print("pass")
    except:
        print("layer is empty")
    layer = ds.GetLayer()
    for feature in layer:
        fieldVal = feature.GetField(field)
        clipRasterPoly(inPathTIF,"data/Optical/Sat/clipFeatures/" + str(fieldVal) + ".shp", "data/Optical/Sat/after/cell_" + str(fieldVal) + ".tif" )

#os.mkdir("data/Optical/Sat/clipFeatures")
#createClipPol("data/Optical/Sat/grid.shp", "id")
clipRasters("data/Optical/Sat/grid.shp", "data/Optical/Sat/DigitalGlobe_14917_modified.tif", "id")