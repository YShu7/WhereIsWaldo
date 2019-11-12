import pickle
import numpy as np
import cyvlfeat as vlfeat
from scipy.spatial.distance import cdist
import cv2
from os.path import join
import matplotlib.image as mpimg


def build_vocab(images, vocab_size):
    dim = 128  # length of the SIFT descriptors that you are going to compute.
    vocab = np.zeros((vocab_size, dim))
    total_SIFT_features = np.zeros((20 * len(images), dim))

    for i, img in enumerate(images):
        frames, descriptors = vlfeat.sift.dsift(img, step=1, fast=True)
        if descriptors.shape[0] > 20:
            idx = np.random.choice(descriptors.shape[0], size=20, replace=False)
        elif descriptors.shape[0] > 0:
            idx = np.random.choice(descriptors.shape[0], size=20, replace=True)
        else:
            continue
        total_SIFT_features[i * 20:(i + 1) * 20] = descriptors[idx, :]
    vocab = vlfeat.kmeans.kmeans(total_SIFT_features, vocab_size)

    return vocab


def bags_of_sifts(images, vocab_filename):
    with open(vocab_filename, 'rb') as f:
        vocab = pickle.load(f)

    feats = np.zeros(shape=(len(images), vocab.shape[0]))
    for idx, img in enumerate(images):
        frames, descriptors = vlfeat.sift.dsift(img, step=20, fast=True)
        D = cdist(descriptors, vocab)
        feature = [0] * vocab.shape[0]
        for d in D:
            feature[np.argmin(d)] += 1
        feature = np.asarray(feature)
        if np.linalg.norm(feature) != 0:
            feature = feature / np.linalg.norm(feature)
        feats[idx] = feature

    return feats


def bags_of_sifts_spm(imgs, vocab_filename, depth):
    with open(vocab_filename, 'rb') as f:
        vocab = pickle.load(f)

    vocab_size = vocab.shape[0]
    feats = []

    # compute the total num of cells in all levels
    num_cell = 0
    for level in range(depth):
        num_cell += 4 ** level
    feats = np.zeros(shape=(len(imgs), vocab_size * num_cell))

    for idx, img in enumerate(imgs):

        this_feature = []
        for level in range(depth):
            cell_per_line = 2 ** level
            width, height, weight = getLevelInfo(img, level, depth)
            # print('level: {0} height: {1} width: {2}'.format(level, width, height))

            for index in range(cell_per_line ** 2):
                min_x = (index // cell_per_line) * width
                max_x = (index // cell_per_line + 1) * width
                min_y = (index % cell_per_line) * height
                max_y = (index % cell_per_line + 1) * height

                patch = img[min_x:max_x, min_y:max_y]
                frames, descriptors = vlfeat.sift.dsift(patch, step=8, fast=True)
                D = cdist(descriptors, vocab)
                feature = np.zeros(shape=(vocab.shape[0]))
                for d in D:
                    feature[np.argmin(d)] += 1
                for f in feature:
                    this_feature.append(f * weight)

        this_feature = np.asarray(this_feature).flatten()
        if np.linalg.norm(this_feature) != 0:
            feats[idx] = this_feature / np.linalg.norm(this_feature)

    return feats


def getLevelInfo(img, level, depth):
    width = img.shape[0]
    height = img.shape[1]
    weight = 1 / 2 ** (depth - 1)
    if level != 0:
        width = int(width / (2 * level))
        height = int(height / (2 * level))
        weight = 1 / 2 ** (depth - level)

    return width, height, weight


def filter_candidate_sift(image_id, candidates_pos, model, threshold, vocab_filename):
    filtered_candidates_pos = []

    img = mpimg.imread(join("datasets/JPEGImages/{}.jpg".format(image_id)))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    for pos in candidates_pos:
        (x, y, w, h) = pos
        x = int(round(x))
        y = int(round(y))
        w = int(round(w))
        h = int(round(h))
        candidate = img[y:y + h, x:x + w]

        if any(i < 8 for i in [x, y, w, h]):
            continue
        candidate_feats = bags_of_sifts_spm([candidate], vocab_filename, 3)
        pred = model.predict_proba(candidate_feats)[0]
        if pred[0] >= threshold:
            info = [image_id, pred[0]]
            info.extend([x, y, x+w, y+h])
            filtered_candidates_pos.append(info)

    return filtered_candidates_pos


def get_res(candidates, candidates_pos, model, vocab_filename=None):
    waldo_list = []
    wenda_list = []
    wizard_list = []

    waldo_candidates = []
    wenda_candidates = []
    wizard_candidates = []

    import math
    index = 0
    for candidate, pos in zip(candidates, candidates_pos):
        val_image_feat = bags_of_sifts_spm([candidate], vocab_filename, 3)
        y_pred = model.predict(val_image_feat)[0]
        if y_pred == 'waldo':
            waldo_list.append(pos)
            waldo_candidates.append(candidate)
        if y_pred == 'wenda':
            wenda_list.append(pos)
            wenda_candidates.append(candidate)
        if y_pred == 'wizard':
            wizard_list.append(pos)
            wizard_candidates.append(candidate)
        index += 1
        if index % 1000 == 0:
            print("res {0}".format(index))
    return waldo_candidates, waldo_list, wenda_candidates, wenda_list, wizard_candidates, wizard_list

