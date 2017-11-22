import numpy as np
import pandas as pd
import os.path
import scipy
import cv2
import sys
from lxml import etree

from caffeutil import *
from optimizerutil import *
from xmlutil import *

def generateAllFrame(sample_dir, xml_INFO, net, BEV_coord, initParams, predLinFit=None):
    image_dir = os.path.dirname(sample_dir)
    samp = os.path.basename(sample_dir)

    l_image =  os.listdir("%s/%s" % (image_dir,  samp))
    l_image = sorted(l_image)

    coord_3d = BEV_coord['coord_3d']
    coord_6m = BEV_coord['coord_600cm']

    M_RTK = parseInfo(xml_INFO)
    src,jac = cv2.projectPoints(coord_3d, M_RTK['R'],  M_RTK['T'], M_RTK['K'], M_RTK['D'] )
    src = src[src[:,0,:].argsort(axis=0)[:,0],0,:]
    src_6m = cv2.projectPoints(coord_6m, M_RTK['R'],  M_RTK['T'], M_RTK['K'], M_RTK['D'] )[0][0][0]

    dst = np.array([[[10,200], [10,0], [90,0], [90,200]]]).astype(np.float32)
    M    = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    mm = np.dot(M, np.array(list(src_6m) + [1.]))
    xy_trans = (mm / mm[2])[0:2]

    refPos = xy_trans
    l_name = []
    l_nameRaw = []
    l_t_x = []
    l_t_y = []
    l_errors = []

    seedParams = []

    for i in range(len(l_image)):
        inName  = "%s/%s/%s" % (image_dir,samp,l_image[i])
        caffeOutImg,rawImg = roadLaneFromSegNet(inName, net)

        rawImg2 = rawImg.copy()
        binarizedImg = caffeOutImg
        binarizedImg_t = cv2.warpPerspective(binarizedImg, M, (100, 200), cv2.WARP_INVERSE_MAP)


        paramSearch = None
        if len(seedParams) == 0:
            paramSearch = initParams
        else:
            lla = np.linspace(seedParams[0]-(initParams[0][1]-initParams[0][0]),
                              seedParams[0]+(initParams[0][1]-initParams[0][0]), 3)
            llb = np.linspace(seedParams[1]-(initParams[1][1]-initParams[1][0]),
                              seedParams[1]+(initParams[1][1]-initParams[1][0]), 3)
            llc = np.arange(seedParams[2]-2, seedParams[2]+3, 2)
            lld = np.arange(seedParams[3]-1, seedParams[3]+2)
            paramSearch = [lla, llb, llc, lld]

        coords,params = getBestParams(binarizedImg_t, paramSearch, refPos)
        #print(params)
        d = params[3]
        seedParams = params
        d = PrevFilter(d, l_errors)
        l_errors.append(d)

    if predLinFit is None:
        return l_errors

    else:
        return [e*predLinFit[0]+predLinFit[1] for e in l_errors]


def extractBinarizedImg(imgout,rawimg):
    rawimg = rawimg[:,:,::-1]
    imgBinary = (imgout>100).astype(np.int)

    for i in range(3):
        rawimg[:,:,i] = rawimg[:,:,i]*imgBinary

    return rawimg


def main():
    sample_dir = sys.argv[1]
    xml_INFO = sys.argv[2]
    model   = sys.argv[3]
    weights = sys.argv[4]

    image_dir = os.path.dirname(sample_dir)
    sample = os.path.basename(sample_dir)

    width   = 300
    x_start = 200
    x_end   = 6000
    coord_3d = np.float32([
        [x_start,-width,0], [x_start, width,0],
        [x_end,  -width,0], [x_end,   width,0]
    ]).reshape(-1,3)
    coord_6m = np.array([600., 0., 0.]).reshape(-1,3)
    BEV_coords = {
        "coord_3d":coord_3d,
        "coord_600cm" : coord_6m
    }

    initParams = [
        np.linspace(0.001, 0.02, 10),
        np.array(list(np.linspace(1.5, 50, 10)) + list(np.linspace(-50, 1.5, 10))),
        np.arange( 23, 28, 2),
        np.arange(-15, 15, 2)
    ]
    predLinFit = np.array([ 6.66317499,  1.35541496])

    net = loadNet(model, weights)
    l_err = generateAllFrame(sample_dir, xml_INFO=xml_INFO, net=net, BEV_coord=BEV_coords, initParams=initParams, predLinFit=predLinFit)

    with open("./%s.xml" % (sample), 'w') as f_out:
        f_out.write("<opencv_storage>\n")
        for i,e in enumerate(l_err):
            f_out.write("  <Frame%0*d>%e</Frame%0*d>\n" % (5, 1, e, 5, 1))

        f_out.write("</opencv_storage>")

if __name__ == '__main__':
    main()
