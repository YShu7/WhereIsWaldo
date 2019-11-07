# ====================================================
# @Time    : 2019/9/9 17:24
# @Author  : Xiao Junbin
# @Email   : junbin@comp.nus.edu.sg
# @File    : evaluation.py
# ====================================================

from voc_eval import *
from utils import *

detpath = 'baseline/{}.txt'
annopath = 'datasets/Annotations/{}.xml'
imagesetfile = 'datasets/ImageSets/val.txt'
cachedir = 'cache_anno'

with open(imagesetfile, 'r') as f:
    lines = f.readlines()
imagenames = [x.strip() for x in lines]
for imagename in imagenames:
    print("Evaluating {}".format(imagename))
    test_image(imagename)

classes = ['waldo', 'wenda', 'wizard']
meanAP = 0
for idx, classname in enumerate(classes) :
    rec, prec, ap = voc_eval(detpath, annopath, imagesetfile, classname,
                                            cachedir, ovthresh=0.5, use_07_metric=False)
    meanAP += ap
    print('{}: {}'.format(classname, ap))

print('meanAP: {}'.format(meanAP/len(classes)))