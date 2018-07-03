import os
from osgeo import gdal, ogr

def clipRasterPoly(rasterPath, polyPath, outPath):
    os.system("gdalwarp -q -cutline " + polyPath + " -crop_to_cutline " + "-of GTiff" + rasterPath + " " + outPath)

def createClipPol(inPath, field):
    driverSHP = ogr.GetDriverByName("ESR Shapefile")
    ds = driverSHP.Open(inPath)
    try:
        ds =! NULL
    except:
        print("layer is empty")
    layer = ds.GetLayer()

    sptlRef = layer.GetSpatialRef()

    for feature in layer:
        fieldVal = feature.GetField(field)
        outDs = driverSHP.CreateDataSource("clipFeatures/" + str(fieldVal) + ".shp")
        outLayer = outDs.CreateLayer("clipFeatures/" + str(fieldVal) + ".shp", srs=sptlRef, geom_type = ogr.wkbPolygon)
        outDfn = outLayer.GetLayerDefn()
        inGeom = feature.GetGeometry.Ref()
        outFeat = ogr.Feature(outDfn)
        outFeat.SetGeometry(inGeom)
        outLayer.CreateFeature(outFeat)

def clipRasters(inPathSHP, inPathTIF, field):
    driverSHP = driverSHP = ogr.GetDriverByName("ESR Shapefile")
    ds = driverSHP.Open(inPathSHP)
    try:
        ds =! NULL
    except:
        print("layer is empty")
    layer = ds.GetLayer()
    
    for feature in layer:
        fieldVal = feature.GetField(field)
        damage = feature.GetField("damage")
        if damage == "unknown":
            folder = "unknown"
        elif damage == "destroyed":
            folder = "dest"
        elif damage == "none":
            folder = "no"
        elif damage == "partial":
            folder = "min"
        elif damage == "significant":
            folder = "sig"
        elif damage == NULL:
            folder = "noData"
        clipRasterPoly(inPathTIF,"clipFeatures/" + str(fieldVal) + ".shp", folder + "/cell_" + str(fieldVal) + ".tif" )

os.mkdir("clipFeatures")
createClipPol("Building_BBox.shp", "osm_id")
ClipRasters("Building_BBox.shp", tiff, "osm_id")