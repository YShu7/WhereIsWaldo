from skimage import feature
import matplotlib.image as mpimg
import cv2
from os.path import join
from sklearn.svm import SVC

def train_hog(X, y):
    data = []
    labels = []

    for label, gray in zip(y, X):
        gray = cv2.resize(gray, (100, 100))

        H = feature.hog(gray, orientations=9, pixels_per_cell=(10, 10),
                        cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2-Hys")

        data.append(H)
        labels.append(label)

    print("[INFO] training classifier...")
    model = SVC(gamma="scale", decision_function_shape='ovo', probability=True, kernel="linear")
    model.fit(data, labels)
    print("[INFO] evaluating...")
    return model

def filter_candidate_hog(image_id, candidates_pos, model, threshold):
    filtered_candidates_pos = []

    img = mpimg.imread(join("datasets/JPEGImages/{}.jpg".format(image_id)))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    for pos in candidates_pos:
        (x,y,w,h) = pos
        x = int(round(x))
        y = int(round(y))
        w = int(round(w))
        h = int(round(h))

        candidate = img[y:y+h, x:x+w]
        new_candidate = cv2.resize(candidate, (100, 100))
        (H, hogImage) = feature.hog(new_candidate, orientations=9, pixels_per_cell=(10, 10),
                                    cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2-Hys", visualize=True)
        pred = model.predict_proba(H.reshape(1, -1))[0]

        if pred[0] >= threshold:
            info = [image_id, pred[0]]
            info.extend(pos)
            filtered_candidates_pos.append(info)

    return filtered_candidates_pos


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
