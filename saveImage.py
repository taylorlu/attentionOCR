# encoding=utf8
from __future__ import print_function
import cv2
import json
import os
import numpy as np
import pickle

# def each_word(anno):
#     for block in anno['annotations']:
#         xx, yy = [], []
#         s = []
#         for char in block:
#             for xy in char['polygon']:
#                 xx.append(xy[0])
#                 yy.append(xy[1])
#             if char['is_chinese']:
#                 s.append(char['text'])
#         yield (min(xx), min(yy), max(xx) - min(xx), max(yy) - min(yy)), s

# with open('word2index.pkl', 'rb') as f:
#     word2index = pickle.load(f)

# print(word2index)

print(u'\u7eb1')

# with open('word2index2.pkl', 'w') as f:
#     pickle.dump()

# a = {'test': '\u4e08'}
# print(a['test'])
'''

labelFiles = 'labels.txt'
labelFP = open(labelFiles, 'w')

idx = 0
with open(r'/data/dataset/CTWDataset/annotate/train.jsonl', encoding='utf-8') as f:
    for line in f:

        line = line.replace('\\u', 'u')
        anno = json.loads(line)

        path = os.path.join(r'/data/dataset/CTWDataset/trainImages', anno['file_name'])

        print('{}, {}'.format(idx, path))
        
        img = cv2.imread(path)
        
        fixBorder = 224

        for bbox, s in each_word(anno):

            bbox = np.array(bbox).astype(np.int32)

            left = int(bbox[1])
            top = int(bbox[0])
            right = int(bbox[1]+bbox[3])
            bottom = int(bbox[0]+bbox[2])

            if(bbox[1]<0):
                left = 0
            if(bbox[0]<0):
                top = 0
            if(right>img.shape[1]):
                right = img.shape[1]
            if(bottom>img.shape[0]):
                bottom = img.shape[0]

            cropImg = img[left:right, top:bottom]

            if(bbox[2]>bbox[3]):
                scale = float(fixBorder)/bbox[2]
            else:
                scale = float(fixBorder)/bbox[3]
            cropImg = cv2.resize(cropImg, (int(cropImg.shape[1]*scale), int(cropImg.shape[0]*scale)))

            tmpImg = np.zeros([fixBorder, fixBorder, 3], dtype=np.int32)
            tmpImg[:cropImg.shape[0], :cropImg.shape[1], :] = cropImg

            name = '/data/dataset/CTWDataset/cropImages/{}.jpg'.format(idx)
            cv2.imwrite(name, tmpImg)
            labelFP.write('{} {}\r\n'.format(name, ",".join(s)))

            idx += 1




fixBorder = 224
cropImg = cv2.imread(r'20190521152746.jpg')

if(cropImg.shape[0]>cropImg.shape[1]):
    scale = float(fixBorder)/cropImg.shape[0]
else:
    scale = float(fixBorder)/cropImg.shape[1]
cropImg = cv2.resize(cropImg, (int(cropImg.shape[1]*scale), int(cropImg.shape[0]*scale)))

tmpImg = np.zeros([fixBorder, fixBorder, 3], dtype=np.int32)
tmpImg[:cropImg.shape[0], :cropImg.shape[1], :] = cropImg

name = 'test.jpg'
cv2.imwrite(name, tmpImg)
'''