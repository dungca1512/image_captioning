import os
import csv
from shutil import copy
from pycocotools.coco import COCO


val_dir = "./data/images/val/"
train_dir = "/kaggle/input/coco2014/train2014/train2014/"
train_ids_csv_orig = "./TrainImageIdsOriginal.csv"
train_ids_csv = "./TrainImageIds.csv"
val_ids_csv = "./ValImageIds.csv"

split_size = 0.8
coco = COCO('/kaggle/input/coco2014/captions/annotations/captions_train2014.json')

if not os.path.isdir(val_dir):
    os.makedirs(val_dir)

with open(train_ids_csv_orig, "r", newline='') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',')
    img_ids = list(csvreader)
img_ids = [int(i) for i in img_ids[0]]

# print(img_ids)

split_id = int(split_size*len(img_ids))
train_ids = img_ids[0:split_id]
val_ids = img_ids[split_id:]

with open(val_ids_csv, "w", newline='') as csvfile:
    csvwriter = csv.writer(csvfile, dialect='excel')
    csvwriter.writerow(val_ids)

with open(train_ids_csv, "w", newline='') as csvfile:
    csvwriter = csv.writer(csvfile, dialect='excel')
    csvwriter.writerow(train_ids)
    
for idx in val_ids:
    path = coco.loadImgs(idx)[0]['file_name']
    copy(train_dir + path, val_dir + path)
