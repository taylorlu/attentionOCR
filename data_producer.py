from __future__ import division
from __future__ import print_function
import json
import numpy as np
import os
import cv2
import random
import pickle


class DataProducer(object):

    def __init__(self, data_json, batch_size, max_step):
        """
        Args:
            data_json : json format file with ocr data.
                        'path', the path of the image file.
                        'label', the char's identity in int.
            batch_size : Size of the batches for training.
            max_step: max label count of the whole dataset.
        """
        self.batch_size = batch_size
        self.max_step = max_step
        self.batch_index = 0
        with open(data_json, 'r') as fid:
            self.data = json.load(fid)
        self.shuffle_samples()


    def shuffle_samples(self):
        random.shuffle(self.data)
        self.batches = [self.data[i:i+self.batch_size]
                   for i in range(0, len(self.data) - self.batch_size + 1, self.batch_size)]

    def get_batch_count(self):
        return len(self.batches)

    def get_batch_sample(self):

        if(self.batch_index>len(self.batches)-1):
            self.shuffle_samples()
            self.batch_index = 0

        sentences = []
        images = []
        masks = []
        for path_label in self.batches[self.batch_index]:
            chars = list(map(int, path_label['label'].split(',')))
            num_chars = len(chars)

            mask = np.zeros(self.max_step, dtype = np.float32)
            mask[:num_chars+1] = 1.0
            masks.append(mask)
            sentence = np.zeros(self.max_step, dtype = np.int32)
            sentence[:num_chars] = chars
            sentence[num_chars] = 3581
            sentences.append(sentence)

            img = cv2.imread(path_label['path'])
            images.append(img)

        self.batch_index += 1
        return images, sentences, masks

'''
with open('word2index.pkl', 'rb') as f:
    word2index = pickle.load(f)


path_labels = []
fp = open('labels.txt', 'r')
for line in fp:
    line = line.strip()
    if(len(line.split(' '))<2):
        continue
    path = line.split(' ')[0]
    _unicodes = line.split(' ')[1]

    values = []
    for uni in _unicodes.split(','):
        values.append(str(word2index[uni]))

    path_label = {}
    path_label['path'] = path
    path_label['label'] = ','.join(values)
    path_labels.append(path_label)

fp2 = open('data.json', 'w')
json.dump(path_labels, fp2)


provider = DataProducer('data.json', 2, 20)
for i in range(5):

    _, sentences, masks = provider.get_batch_sample()
    print(sentences)
'''