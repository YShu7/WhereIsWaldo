import pickle
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.image as mpimg
import numpy as np
from image_generator import *
from sift_utils import *
from hog_utils import *
import time
import cv2
import os.path as osp
from sklearn.svm import SVC
from skimage import feature


def save(model, filename):
    with open("model/{0}.pkl".format(filename), 'wb') as f:
        pickle.dump(model, f)
        print('{:s} saved'.format(filename))


def load(filename):
    with open("model/{0}.pkl".format(filename), 'rb') as f:
        model = pickle.load(f)
    return model


def write(directory, waldo_list, wenda_list, wizard_list):
    """
    Write output to file
    """
    with open('{}/waldo.txt'.format(directory), 'a') as f:
        for item in waldo_list:
            for p in item:
                f.write("%s " % p)
            f.write("\n")

    with open('{}/wenda.txt'.format(directory), 'a') as f:
        for item in wenda_list:
            for p in item:
                f.write("%s " % p)
            f.write("\n")

    with open('{}/wizard.txt'.format(directory), 'a') as f:
        for item in wizard_list:
            for p in item:
                f.write("%s " % p)
            f.write("\n")

def validate_body(image_id, faces, filename, wt=2, ht=3, s=1.3, mn=3, t=0):
    """
    Validate if the faces detected is from an character with body shown in the image.
    """
    win = []
    win_f = []
    img = mpimg.imread(join("datasets/JPEGImages/{}.jpg".format(image_id)))
    for pos in faces:
        (_, _, x_min, y_min, x_max, y_max) = pos
        w = int(round(x_max-x_min))
        h = int(round(y_max-y_min))
        ow = w
        oh = h
        w = int(round(w*wt))
        h = int(round(h*ht))
        x = int(round(x_min - w/2 + ow/2))
        y = int(round(y_min))

        candidate = img[y:y+h, x:x+w]

        waldo_body_cascade = cv2.CascadeClassifier(filename)
        waldo_bodies, _, waldo_bscore = waldo_body_cascade.detectMultiScale3(
            candidate, 
            scaleFactor=s, 
            minNeighbors=mn, 
            minSize=(w//wt, h//ht), 
            flags=cv2.CASCADE_SCALE_IMAGE, 
            outputRejectLevels=True)
        waldo_bodies = [waldo_body for i, waldo_body in enumerate(waldo_bodies) if waldo_bscore[i][0] > t]
        waldo_bscore = [score for i, score in enumerate(waldo_bscore) if waldo_bscore[i][0] > t]
        if len(waldo_bodies) == 0:
            win_f.append(pos)
        for i, b in enumerate(waldo_bodies):
            bx, by, bw, bh = b
            x_min = min(x_min, bx+x)
            y_min = min(y_min, by+y)
            win.append([image_id, waldo_bscore[i][0], x_min, y_min, x_min+bw, y_min+bh*1.3])
    return win, win_f


def suppress(bodies, threshold=0.9):
    """
    Suppress bodies if any of them share ratio of overlap threshold > 0.9.
    """
    if len(bodies) == 0:
        return []
    boxes = np.stack(bodies, axis=0)
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 2].astype(float)
    y1 = boxes[:, 3].astype(float)
    x2 = boxes[:, 4].astype(float)
    y2 = boxes[:, 5].astype(float)

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > threshold)[0])))

    return boxes[pick].tolist()


def merge(bodies, faces):
    """
    Remove bodies which are part of identified bodies.
    :param bodies: (image
    :param faces:
    :return:
    """
    merged_faces = []
    for face in faces:
        _, _, x21, y21, x22, y22 = face
        should_merge = True
        for body in bodies:
            _, _, x11, y11, x12, y12 = body
            overlap_x1 = max(x11, x21)
            overlap_x2 = min(x12, x22)
            overlap_y1 = max(y11, y21)
            overlap_y2 = min(y12, y22)
            if overlap_x2 < overlap_x1 or overlap_y2 < overlap_y1:
                should_merge = True
            else:
                overlap = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                union = (y22 - y21) * (x22 - x21) + (y12 - y11) * (x12 - x11) - overlap
                score = overlap / union
                if score < 0.1:
                    should_merge = True
                else:
                    should_merge = False
                    break
        if should_merge:
            merged_faces.append(face)
    return merged_faces

def test_image(image_id, path):
    img = mpimg.imread(join("datasets/JPEGImages/{}.jpg".format(image_id)))
    min_width = int(round(img.shape[0] ** 0.01))
    min_height = int(round(min_width * 5 / 4))
    waldo_face_cascade = cv2.CascadeClassifier("xml/waldo_40_40.xml")
    waldo_faces, _, waldo_score = waldo_face_cascade.detectMultiScale3(
        img,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(min_width, min_width),
        flags = cv2.CASCADE_SCALE_IMAGE,
        outputRejectLevels = True
    )
    waldo_faces = [waldo_face for i, waldo_face in enumerate(waldo_faces) if waldo_score[i][0] > 3]
    waldo = load("hog_waldo")
    waldo_faces = filter_candidate_hog(image_id, waldo_faces, waldo, 0.15)
    waldo_bodies, waldo_faces = validate_body(image_id, waldo_faces, "xml/waldo_body_0.3_0.0002.xml", wt=3, ht=4, mn=5, t=1)
    waldo_faces.sort(key=lambda x: x[1], reverse=True)
    waldo_faces = waldo_faces[:5]

    wenda_face_cascade = cv2.CascadeClassifier("xml/wenda_0.5_0.0007.xml")
    wenda_faces, _, wenda_score = wenda_face_cascade.detectMultiScale3(
        img,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(min_width, min_height),
        flags = cv2.CASCADE_SCALE_IMAGE,
        outputRejectLevels = True
    )
    wenda_faces = [wenda_face for i, wenda_face in enumerate(wenda_faces) if wenda_score[i][0] > 2]
    wenda = load("hog_wenda")
    wenda_faces = filter_candidate_hog(image_id, wenda_faces, wenda, 0.1)
    wenda_bodies, wenda_faces = validate_body(image_id, wenda_faces, "xml/wenda_body_0.3_0.0002.xml", mn=2, t=0)
    wenda_faces.sort(key=lambda x: x[1], reverse=True)
    wenda_faces = wenda_faces[:5]

    wenda_faces = merge(waldo_bodies, wenda_faces)
    waldo_faces = merge(wenda_bodies, waldo_faces)

    wizard_face_cascade = cv2.CascadeClassifier("xml/wizard_0.3_3e-5.xml")
    wizard_faces, _, wizard_score = wizard_face_cascade.detectMultiScale3(
        img,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(min_width, min_height),
        flags = cv2.CASCADE_SCALE_IMAGE,
        outputRejectLevels = True
    )
    wizard_faces = [wizard_face for i, wizard_face in enumerate(wizard_faces) if wizard_score[i][0] > 2]
    wizard = load("hog_wizard")
    wizard_faces = filter_candidate_hog(image_id, wizard_faces, wizard, 0.2)
    wizard_bodies, wizard_faces = validate_body(image_id, wizard_faces, "xml/wizard_body_0.0003.xml", wt=3, ht=4, mn=4, t=0)
    wizard_faces.sort(key=lambda x: x[1], reverse=True)
    wizard_faces = wizard_faces[:5]

    wenda_bodies = suppress(wenda_bodies)
    waldo_bodies = suppress(waldo_bodies)
    wizard_bodies = suppress(wizard_bodies)

    waldo_bodies.extend(waldo_faces)
    wenda_bodies.extend(wenda_faces)
    wizard_bodies.extend(wizard_faces)
    write(path, waldo_bodies, wenda_bodies, wizard_bodies)
    return waldo_bodies, wenda_bodies, wizard_bodies


def show_svm_res(f, val_images_all, vocab_size, training_set_this, train_labels_this, val_labels_this, categories, ax):
    vocab_filename = "vocab/{0}_{1}.pkl".format(f, vocab_size)
    if not osp.isfile(vocab_filename):
        print('No existing visual word vocabulary found. Computing one from training images')
        vocab = build_vocab(training_set_this, vocab_size)

        with open(vocab_filename, 'wb') as f:
            pickle.dump(vocab, f)
            print('{:s} saved'.format(vocab_filename))

    train_image_feats = bags_of_sifts_spm(training_set_this, vocab_filename, 3)
    svm = SVC(gamma="scale", decision_function_shape='ovo', probability=True, kernel="linear")
    svm.fit(train_image_feats, train_labels_this)

    start_time = time.time()
    val_image_feats = bags_of_sifts_spm(val_images_all, vocab_filename, 3)
    y_pred = svm.predict(val_image_feats)
    print(time.time() - start_time)

    show_results(val_labels_this, categories, y_pred, ax)
    return svm


def show_hog_res(val_images_all, val_labels_this, model, ax, categories=["waldo", "wenda", "wizard", "others"]):
    preds = []
    for img in val_images_all:
        new_candidate = cv2.resize(img, (100, 100))
        (H, hogImage) = feature.hog(new_candidate, orientations=9, pixels_per_cell=(10, 10),
                                    cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2-Hys", visualize=True)
        pred = model.predict(H.reshape(1, -1))[0]
        preds.append(pred)

    show_results(val_labels_this, categories, preds, ax)


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