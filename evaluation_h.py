# ====================================================
# @Time    : 2019/9/9 17:24
# @Author  : Xiao Junbin
# @Email   : junbin@comp.nus.edu.sg
# @File    : evaluation.py
# ====================================================

from voc_eval import *
from utils import *

detpath = 'baseline_1/{}.txt'
annopath = 'datasets/Annotations/{}.xml'
imagesetfile = 'datasets/ImageSets/val.txt'
cachedir = 'cache_anno'

for the_file in os.listdir("baseline_1"):
    file_path = os.path.join("baseline_1", the_file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
    except Exception as e:
        print(e)

with open(imagesetfile, 'r') as f:
    lines = f.readlines()
imagenames = [x.strip() for x in lines]
for imagename in imagenames:
    print("Evaluating {}".format(imagename))
    test_image(imagename, "baseline_1")

classes = ['waldo', 'wenda', 'wizard']
meanAP = 0
for idx, classname in enumerate(classes) :
    rec, prec, ap = voc_eval(detpath, annopath, imagesetfile, classname,
                                            cachedir, ovthresh=0.3, use_07_metric=False)
    meanAP += ap
    print('{}: {}'.format(classname, ap))

print('meanAP: {}'.format(meanAP/len(classes)))