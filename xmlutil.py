import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path
import scipy
import cv2
import sys
import re
from lxml import etree



def parseChildren(dataSub):
    dim_x = int(dataSub.getchildren()[0].text)
    dim_y = int(dataSub.getchildren()[1].text)
    return np.array(
        filter(
            lambda x: len(x)>0, re.split("\n|\s+", dataSub.getchildren()[-1].text)
        )).astype(np.float64).reshape([dim_x, dim_y])

def parseInfo(xml_INFO):
    tree = etree.parse(xml_INFO)
    data = tree.getroot()
    M_RTK = {}
    for node in data:
        if node.tag == "rotation_matrix":
            M_RTK['R'] = parseChildren(node)
        if node.tag == "translation_vector":
            M_RTK['T'] = parseChildren(node)
        if node.tag == "camera_matrix":
            M_RTK['K'] = parseChildren(node)
        if node.tag == "distortion_coefficients":
            M_RTK['D'] = parseChildren(node)

    return M_RTK
