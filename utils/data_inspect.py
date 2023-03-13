from __future__ import print_function
import numpy as np
import utils.wsad_utils as utils
import random
import os
import options
from collections import Counter


class SampleDataset:
    def __init__(self, mode="both", sampling='random'):
        self.dataset_name = 'Thumos14reduced'
        self.num_class = 20
        self.sampling = sampling
        self.num_segments = 320
        self.feature_size = 2048
        path_dataset = '/dev/THUMOS14/Thumos14reduced'

        self.path_to_features = os.path.join(path_dataset, self.dataset_name + "-I3D-JOINTFeatures.npy")
        self.path_to_annotations = os.path.join(path_dataset, self.dataset_name + "-Annotations/")
        self.features = np.load(
            self.path_to_features, encoding="bytes", allow_pickle=True
        )
        self.segments = np.load(
            self.path_to_annotations + "segments.npy", allow_pickle=True
        )
        self.labels = np.load(
            self.path_to_annotations + "labels_all.npy", allow_pickle=True
        )
        # Specific to Thumos14

        self._labels = np.load(
            self.path_to_annotations + "labels.npy", allow_pickle=True
        )
        self.classlist = np.load(
            self.path_to_annotations + "classlist.npy", allow_pickle=True
        )
        self.subset = np.load(
            self.path_to_annotations + "subset.npy", allow_pickle=True
        )
        self.videonames = np.load(
            self.path_to_annotations + "videoname.npy", allow_pickle=True
        )
        self.batch_size = 10
        self.trainidx = []
        self.testidx = []
        self.classwiseidx = []
        self.currenttestidx = 0
        self.labels_multihot = [
            utils.strlist2multihot(labs, self.classlist)
            for labs in self.labels
        ]
        try:
            ambilist = self.path_to_annotations + "/Ambiguous_test.txt"
            ambilist = list(open(ambilist, "r"))
            ambilist = [a.strip("\n").split(" ")[0] for a in ambilist]
        except:
            ambilist = []
        self.train_test_idx()
        self.classwise_feature_mapping()

        self.normalize = False
        self.mode = mode
        if mode == "rgb" or mode == "flow":
            self.feature_size = 1024

    def train_test_idx(self):
        for i, s in enumerate(self.subset):

            if s.decode("utf-8") == "validation":  # Specific to Thumos14

                self.trainidx.append(i)
            elif s.decode("utf-8") == "test":
                self.testidx.append(i)

    def classwise_feature_mapping(self):
        for category in self.classlist:
            idx = []
            for i in self.trainidx:
                for label in self.labels[i]:
                    if label == category.decode("utf-8"):
                        idx.append(i)
                        break
            self.classwiseidx.append(idx)


if __name__ == '__main__':

    data = SampleDataset()
    # print(data)
    print('total video number:{}'.format(data.features.shape))
    print('train video number:{}'.format(len(data.trainidx)))
    print('test video number:{}'.format(len(data.testidx)))
    seq_lengths = [i.shape[0] for i in data.features]
    seq_lengths = sorted(seq_lengths)
    min_ = seq_lengths[0]
    max_ = seq_lengths[-1]
    media_ = seq_lengths[int(len(seq_lengths)/2)]
    print('min length:{}'.format(min_))
    print('max_ length:{}'.format(max_))
    print('media_ length:{}'.format(media_))
    temp = [i for i in seq_lengths if i < 320]
    print('length less than 320 ratio:{}'.format(len(temp)/len(seq_lengths)))
    temp = [i for i in seq_lengths if i < 500]
    print('length less than 500 ratio:{}'.format(len(temp)/len(seq_lengths)))

    cnt = 0
    for idx, v in enumerate(data.classwiseidx):
        print('training set: class {} has {} videos'.format(idx, len(v)))
        cnt += len(v)
    cls_cnt = [sum(i) for i in data.labels_multihot]
    count = Counter(cls_cnt)
    for k, v in count.items():
        print('{} video has {} label'.format(v, int(k)))
    for i in range(3):
        temp_features = data.features[i]
        temp_labels_multihot = data.labels_multihot[i]
        print(temp_features.shape)      
        print(temp_labels_multihot.shape)      