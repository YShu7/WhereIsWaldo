
def gen_seq(start,end,interval):
    result = []
    while(start<end):
        result.append(start)
        start += interval
    return result


def slide_window(img):
    sliding_window_x = 0.005
    sliding_window_y = 0.01
    window_size_x = 0.025
    window_size_y = 0.05


    import math
    candidates = []
    for i in gen_seq(0, img.shape[0] * (1 - 1.01*window_size_y), sliding_window_y*img.shape[0]):
        for j in gen_seq(0, img.shape[1] * (1 - 1.01*window_size_x), sliding_window_x*(img.shape[1])):
            candidate = img[math.floor(i):math.floor(i + window_size_y * img.shape[0]),math.floor(j): math.floor(j + window_size_x * img.shape[1])]
            candidates.append(candidate)
    return candidates