import os
from os.path import isfile, join
import xml.etree.ElementTree as ET
import matplotlib.image as mpimg
import numpy as np
import cv2
import imageio

def get_train_val():
    """
    Get train and validation image ids from train.txt and val.txt

    :return train: n list, where n is the number of the ids of training images
    :return val: m list, where m is the number of the ids of training images
    """
    directory = "datasets/ImageSets"
    train = "train.txt"
    val = "val.txt"
    with open(join(directory, train), 'r') as file_train:
        train = file_train.read().splitlines()
    with open(join(directory, val), 'r') as file_val:
        val = file_val.read().splitlines()
    return train, val


def get_annotation(image_id: str, target='waldo'):
    """
    Read annotation of image and target.
    :param image_id: image id to be read
    :param target: target character
    
    :return n x 4 matrix, where n is the number of candidates' annotation
    """
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


def get_character_image(image_id: str):
    """
    Get character images of the image with input id.
    """
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


def generate_character_images(train_image_ids):
    """
    Get character images of all the images with input ids.
    """
    train_images = {"waldo": [], "wenda": [], "wizard": []}
    for image_id in train_image_ids:
        image_map = get_character_image(image_id)
        for target in image_map:
            for image in image_map[target]:
                train_images[target].append(image)

    return train_images


def get_images(image_ids):
    """
    Get all gray_scale images with input id
    :param image_ids: ids of images to be loaded
    :return:
    """
    val_images = []
    for image_id in image_ids:
        directory = "datasets/JPEGImages"

        img = mpimg.imread(join(directory, '{}.jpg'.format(image_id)))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        val_images.append(np.asarray(img))

    return val_images


def load_train_images(directory, isGray=False):
    """
    Get all gray_scale images of the input directory
    :param directory: the directory that images are loaded from
    :return:
    """
    train_images = []

    paths = [f for f in os.listdir(directory) if isfile(join(directory, f)) and f.endswith('.jpg')]
    for path in paths:
        img = mpimg.imread(join(directory, path))
        
        if not isGray:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        train_images.append(np.asarray(img))
    return train_images


def generate_others(images, ratio):
    """
    Generate other characters (negative images) randomly
    :param images: source images that negative images are cropped from
    """

    for t, img in enumerate(images):
        for id in range(10):
            h = int(round(img.shape[0] * ratio))
            x = np.random.randint(h, img.shape[0] - h)
            y = np.random.randint(h, img.shape[1] - h)

            imageio.imwrite('training_set/others/{}.jpg'.format(t * 10 + id), img[x:x + h - 1, y:y + h - 1])