from qgis.utils import iface
from PyQt4.QtCore import *
from qgis.analysis import QgsZonalStatistics
import time

layers = iface.legendInterface().layers()

polygonLayer = layers[0]
count = 1
for layer in layers[1:]:
    print(layer.name())
    zoneStat = QgsZonalStatistics(polygonLayer,layer.source(),str(count)+ "_", 1, QgsZonalStatistics.Count|QgsZonalStatistics.Mean|QgsZonalStatistics.Median|QgsZonalStatistics.StDev|QgsZonalStatistics.Min|QgsZonalStatistics.Max)
    zoneStat.calculateStatistics(None)
    time.sleep(180)
    count += 1