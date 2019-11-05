import os
from os.path import isfile, join
from scipy.spatial.distance import cdist
import xml.etree.ElementTree as ET
import matplotlib.image as mpimg
import numpy as np
import cv2
import cyvlfeat as vlfeat
import pickle
from skimage import feature
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def get_train_val():
    directory = "datasets/ImageSets"
    train = "train.txt"
    val = "val.txt"
    with open(join(directory, train), 'r') as file_train:
        train = file_train.read().splitlines()
    with open(join(directory, val), 'r') as file_val:
        val = file_val.read().splitlines()
    return train, val

def get_annotation(image_id: str, target='waldo'):
    directory = "datasets/Annotations"

    tree = ET.parse(join(directory, '{}.xml'.format(image_id)))
    root = tree.getroot()

    annos = []

    for boxes in root.iter('object'):

        name = boxes.find('name').text

        if target != name:
            continue

        ymin, xmin, ymax, xmax = None, None, None, None

        for box in boxes.findall("bndbox"):
            ymin = int(box.find("ymin").text)
            xmin = int(box.find("xmin").text)
            ymax = int(box.find("ymax").text)
            xmax = int(box.find("xmax").text)
            annos.append([xmin, ymin, xmax, ymax])

    return annos


def get_image(image_id: str):
    directory = "datasets/JPEGImages"

    img = mpimg.imread(join(directory, '{}.jpg'.format(image_id)))

    targets = ['waldo', 'wizard', 'wenda']
    image_map = {}
    for target in targets:
        annos = get_annotation(image_id, target=target)

        images = []
        for anno in annos:
            images.append(img[anno[1]:anno[3], anno[0]:anno[2]])
        image_map[target] = images

    return image_map

def generate_train_images():
    train_image_ids, val_image_ids = get_train_val()

    train_images = {"waldo": [], "wenda": [], "wizard": []}
    for image_id in train_image_ids:
        image_map = get_image(image_id)
        for target in image_map:
            for image in image_map[target]:
                train_images[target].append(image)

    return train_images


def get_images(image_ids):
    val_images = []
    for image_id in image_ids:
        directory = "datasets/JPEGImages"

        img = mpimg.imread(join(directory, '{}.jpg'.format(image_id)))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        val_images.append(np.asarray(img))

    return val_images


def load_train_images(directory):
    train_images = []

    paths = [f for f in os.listdir(directory) if isfile(join(directory, f)) and f.endswith('.jpg')]
    for path in paths:
        img = mpimg.imread(join(directory, path))
        train_images.append(np.asarray(img))
    return train_images

def build_vocab(images, vocab_size):
    dim = 128  # length of the SIFT descriptors that you are going to compute.
    vocab = np.zeros((vocab_size, dim))
    total_SIFT_features = np.zeros((20 * len(images), dim))

    for i, img in enumerate(images):
        frames, descriptors = vlfeat.sift.dsift(img, step=1, fast=True)
        if descriptors.shape[0] > 20:
            idx = np.random.choice(descriptors.shape[0], size=20, replace=False)
        else:
            idx = np.random.choice(descriptors.shape[0], size=20, replace=True)
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
    """
    Bags of sifts with spatial pyramid matching.

    :param image_paths: paths to N images
    :param vocab_filename: Path to the precomputed vocabulary.
          This function assumes that vocab_filename exists and contains an
          vocab_size x 128 ndarray 'vocab' where each row is a kmeans centroid
          or visual word. This ndarray is saved to disk rather than passed in
          as a parameter to avoid recomputing the vocabulary every run.
    :param depth: Depth L of spatial pyramid. Divide images and compute (sum)
          bags-of-sifts for all image partitions for all pyramid levels.
          Refer to the explanation in the notebook, tutorial slide and the
          original paper (Lazebnik et al. 2006.) for more details.

    :return image_feats: N x d matrix, where d is the dimensionality of the
          feature representation. In this case, d will equal the number of
          clusters (vocab_size) times the number of regions in all pyramid levels,
          which is 21 (1+4+16) in this specific case.
    """
    with open(vocab_filename, 'rb') as f:
        vocab = pickle.load(f)

    vocab_size = vocab.shape[0]
    feats = []

    #############################################################################
    # TODO: YOUR CODE HERE                                                      #
    #############################################################################

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
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

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


def gen_seq(start,end,interval):
    result = []
    while(start<end):
        result.append(start)
        start += interval
    return result

def slide_window(image_id):
    directory = "datasets/JPEGImages"

    img = mpimg.imread(join(directory, '{}.jpg'.format(image_id)))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    sliding_window_x = 0.005
    sliding_window_y = 0.005
    window_size_x = 0.025
    window_size_y = 0.025

    import math
    candidates = []
    candidates_pos = []
    for i in gen_seq(0, img.shape[0] * (1 - 1.01 * window_size_y), sliding_window_y * img.shape[0]):
        for j in gen_seq(0, img.shape[1] * (1 - 1.01 * window_size_x), sliding_window_x * (img.shape[1])):
            candidate = img[math.floor(i):math.floor(i + window_size_y * img.shape[0]),
                        math.floor(j): math.floor(j + window_size_x * img.shape[1])]
            candidates.append(candidate)
            candidates_pos.append([math.floor(i), math.floor(i + window_size_y * img.shape[0]),
                        math.floor(j), math.floor(j + window_size_x * img.shape[1])])
    return candidates, candidates_pos

def filter_candidate(candidates, candidates_pos, model, threshold):
    filtered_candidates = []
    filtered_candidates_pos = []

    index = 0
    for candidate, pos in zip(candidates, candidates_pos):
        new_candidate = cv2.resize(candidate, (100, 100))
        (H, hogImage) = feature.hog(new_candidate, orientations=9, pixels_per_cell=(10, 10),
                                    cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2-Hys", visualize=True)
        pred = model.predict_proba(H.reshape(1, -1))[0]

        if pred[0] >= threshold:
            filtered_candidates.append(candidate)
            filtered_candidates_pos.append(pos)

        index += 1
        if index % 5000 == 0:
            print("filtered {0} to {1}".format(index, len(filtered_candidates)))
    return filtered_candidates, filtered_candidates_pos

def filter_candidate_sift(candidates, candidates_pos, model, vocab_filename):
    waldo_can_all, waldo_list_all, wenda_can_all, wenda_list_all, wizard_can_all, wizard_list_all = get_res(candidates, candidates_pos, model, vocab_filename=vocab_filename)
    fcandidates = []
    fcandidates.extend(waldo_can_all)
    fcandidates.extend(wenda_can_all)
    fcandidates.extend(wizard_can_all)
    fcandidates_pos = []
    fcandidates_pos.extend(waldo_list_all)
    fcandidates_pos.extend(wenda_list_all)
    fcandidates_pos.extend(wizard_list_all)
    return fcandidates, fcandidates_pos

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

def get_res_hog(candidates, candidates_pos, model):
    waldo_list = []
    wenda_list = []
    wizard_list = []

    waldo_candidates = []
    wenda_candidates = []
    wizard_candidates = []

    import math
    index = 0
    for candidate, pos in zip(candidates, candidates_pos):
        new_candidate = cv2.resize(candidate, (100, 100))
        (H, hogImage) = feature.hog(new_candidate, orientations=9, pixels_per_cell=(10, 10),
                                    cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2-Hys",
                                    visualize=True)
        y_pred = model.predict(H.reshape(1, -1))[0]
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


def show_results(test_labels, categories, predicted_categories, ax, cmap=plt.cm.Blues):
    cat2idx = {cat: idx for idx, cat in enumerate(categories)}

    # confusion matrix
    y_true = [cat2idx[cat] for cat in test_labels]
    y_pred = [cat2idx[cat] for cat in predicted_categories]
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype(np.float) / cm.sum(axis=1)[:, np.newaxis]
    acc = np.mean(np.diag(cm))
    print(f'Average Accuracy: {acc*100:.2f}%')

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=categories, yticklabels=categories,
           title='Mean of diagonal = {:4.2f}%'.format(acc*100),
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="black")