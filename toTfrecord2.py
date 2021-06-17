#こっちが自分たちで集めてきた画像版.
#先にmain2.pyを実行してね


import tensorflow as tf
import os
import numpy as np
from object_detection.utils import dataset_util
import json
import glob
import cv2
from tfexamplecreater import output_tfrecord

"""
{
 height,
 width,
 filename,
 image_format,
 xmins,xmaxs,
 ymins,
 ymaxs,
 classes_text,
 classes
 }
"""

detectpath = "C:/Users/kosakae256/Documents/01Develop/CollectNumberPlates/03.detectimgs"
formpath = "C:/Users/kosakae256/Documents/01Develop/CollectNumberPlates/02.formimgs"
jsonpath = "C:/Users/kosakae256/Documents/01Develop/CollectNumberPlates/detectjsons"
temp_data_dir = 'C:/Users/kosakae256/Documents/01Develop/CollectNumberPlates/01.tempimgs'


c = {"number": 1,
     "car": 2,
     }

def totfrecord(jsonpath,datanumber):
    json_open = open(jsonpath, 'r')
    js = json.load(json_open)

    imgpath = f"""{formpath}/form{js["filename"]}"""
    detectimgpath = f"""{detectpath}/detect{js["filename"]}"""

    #detectのほうに画像が存在していなかったらスキップ(誤検出検知)
    if os.path.isfile(detectimgpath) == False:
        print(f"""{js["filename"]}は削除済み""")
        return

    img = cv2.imread(imgpath)
    height, width= img.shape[:2]
    filename = js["filename"]
    format = "jpg"
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for box in js["bbox"]:
        xmins.append(box[0])
        ymins.append(box[1])
        xmaxs.append(box[2])
        ymaxs.append(box[3])

    for tex in js["class"]:
        classes_text.append(tex)
        classes.append(c[tex])

    #整形データ
    jsondata = {"format":format,
                "filename":filename,
                "width":width,
                "height":height,
                "xmins":xmins,
                "xmaxs":xmaxs,
                "ymins":ymins,
                "ymaxs":ymaxs,
                "classes_text":classes_text,
                "classes":classes,
    }


    output_tfrecord(jsondata,datanumber)

if "__main__" == __name__:
    plusnum = 1000
    jsonpath = 'C:/Users/kosakae256/Documents/01Develop/CollectNumberPlates/detectjsons'
    jsondata_list = sorted(glob.glob(jsonpath + '/*.json'))
    for filecount, fpath in enumerate(jsondata_list):
        totfrecord(fpath,filecount+plusnum)
