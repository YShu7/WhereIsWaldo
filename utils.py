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

def test_image(image_id):
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.axis('off')

    img = mpimg.imread(join("datasets/JPEGImages/{}.jpg".format(image_id)))

    # 40-40 1.1 21

    waldo_face_cascade = cv2.CascadeClassifier("lbp_40_40.xml")
    waldo_faces = waldo_face_cascade.detectMultiScale(img, 1.1, 21)
    wenda_face_cascade = cv2.CascadeClassifier("wenda_0.5_0.0007.xml")
    wenda_faces = wenda_face_cascade.detectMultiScale(img, 1.1, 21)
    wizard_face_cascade = cv2.CascadeClassifier("wizard_0.3_3e-5.xml")
    wizard_faces = wizard_face_cascade.detectMultiScale(img, 1.05, 5)
    waldo_body_cascade = cv2.CascadeClassifier("waldo_body_0.0003.xml")
    waldo_bodies = waldo_body_cascade.detectMultiScale(img, 1.1, 5)
    wenda_body_cascade = cv2.CascadeClassifier("wenda_body_2e-5.xml")
    wenda_bodies = wenda_body_cascade.detectMultiScale(img, 1.1, 5)
    wizard_body_cascade = cv2.CascadeClassifier("wizard_body_0.0001.xml")
    wizard_bodies = wizard_body_cascade.detectMultiScale(img, 1.1, 5)

    im = ax.imshow(img)
    targets = ["waldo", "wenda", "wizard"]
    for target in targets:
        annos = get_annotation(image_id, target=target)
        for anno in annos:
            x_min, y_min, x_max, y_max = anno
            ax.add_patch(Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1,
                                   edgecolor='w', facecolor=(1, 1, 1, 0.7)))
            ax.text(x_min, y_min, target)

    waldo = load("hog_waldo")
    waldo_faces = filter_candidate_hog(image_id, waldo_faces, waldo, 0.5)
    waldo_faces.sort(key=lambda x: x[1])
    waldo_faces = waldo_faces[:5]

    waldo_body = load("hog_waldo_body")
    waldo_bodies = filter_candidate_hog(image_id, waldo_bodies, waldo_body, 0.5)
    waldo_bodies.sort(key=lambda x: x[1])
    waldo_bodies = waldo_bodies[:5]

    waldo_bboxes = waldo_faces + waldo_bodies
    print(len(waldo_faces), len(waldo_bodies), len(waldo_bboxes))
    for _, score, x, y, w, h in waldo_bboxes:
        # for x, y, w, h in waldo_faces:
        ax.add_patch(Rectangle((x, y), w, h, linewidth=3,
                               edgecolor='r', facecolor=(1, 1, 1, 0.3)))
        ax.text(x, y, 'waldo-{0:.3f}'.format(score))

    wenda = load("svm_wenda")
    wenda_faces = filter_candidate_sift(image_id, wenda_faces, wenda, 0.5, "vocab/vocab_wenda_200.pkl")
    wenda_faces.sort(key=lambda x: x[1])
    wenda_faces = wenda_faces[:5]

    wenda_body = load("hog_wenda_body")
    wenda_bodies = filter_candidate_sift(image_id, wenda_bodies, wenda_body, 0.5, "vocab/vocab_wenda_200.pkl")
    wenda_bodies.sort(key=lambda x: x[1])
    wenda_bodies = wenda_bodies[:5]

    wenda_bboxes = wenda_faces + wenda_bodies
    for _, score, x, y, w, h in wenda_bboxes:
        ax.add_patch(Rectangle((x, y), w, h, linewidth=3,
                               edgecolor='b', facecolor=(1, 1, 1, 0.3)))
        ax.text(x, y + h / 2, 'wenda-{0:.3f}'.format(score))

    wizard = load("hog_wizard")
    wizard_faces = filter_candidate_hog(image_id, wizard_faces, wizard, 1)
    wizard_faces.sort(key=lambda x: x[1])
    wizard_faces = wizard_faces[:5]

    wizard_body = load("hog_waldo_body")
    wizard_bodies = filter_candidate_hog(image_id, wizard_bodies, wizard_body, 0.5)
    wizard_bodies.sort(key=lambda x: x[1])
    wizard_bodies = wizard_bodies[:5]

    wizard_bboxes = wizard_faces + wizard_bodies
    for _, score, x, y, w, h in wizard_bboxes:
        ax.add_patch(Rectangle((x, y), w, h, linewidth=3,
                               edgecolor='g', facecolor=(1, 1, 1, 0.3)))
        ax.text(x, y + h, 'wizard-{0:.3f}'.format(score))

    write("baseline", waldo_bboxes, wenda_bboxes, wizard_bboxes)

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


def plot(image_id, bboxes, color='r'):
    fig, ax = plt.subplots(figsize=(20,16))
    ax.axis('off')
    img = mpimg.imread('datasets/JPEGImages/{}.jpg'.format(image_id))
    im = ax.imshow(img)
    annos_waldo = get_annotation(image_id, target='waldo')
    for anno in annos_waldo:
        x_min, y_min, x_max, y_max = anno
        bbox = ax.add_patch(Rectangle((x_min,y_min),x_max-x_min,y_max-y_min, linewidth=1,
                                      edgecolor='b', facecolor=(1,1,1,0.5)))
    annos_wenda = get_annotation(image_id, target='wenda')
    for anno in annos_wenda:
        x_min, y_min, x_max, y_max = anno
        bbox = ax.add_patch(Rectangle((x_min,y_min),x_max-x_min,y_max-y_min, linewidth=1,
                                      edgecolor='b', facecolor=(1,1,1,0.5)))
    annos_wizard = get_annotation(image_id, target='wizard')
    for anno in annos_wizard:
        x_min, y_min, x_max, y_max = anno
        bbox = ax.add_patch(Rectangle((x_min,y_min),x_max-x_min,y_max-y_min, linewidth=1,
                                      edgecolor='b', facecolor=(1,1,1,0.5)))

    for box in bboxes:
        y_min, y_max, x_min, x_max = box
        bbox = ax.add_patch(Rectangle((x_min,y_min),x_max-x_min,y_max-y_min, linewidth=1,
                                  edgecolor=color, facecolor='none'))