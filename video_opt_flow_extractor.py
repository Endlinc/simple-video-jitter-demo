from __future__ import print_function

import numpy as np
import cv2 as cv

import video


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:, np.newaxis]
    res = cv.remap(img, flow, None, cv.INTER_LINEAR)
    return res


def main():
    file_path = "/Users/ming/Desktop/IMG_5789.MOV"

    cam = video.create_capture(file_path)
    # read video content into a buffer
    _, prev = cam.read()
    prev = prev[int(prev.shape[0] / 4):int(prev.shape[0] / 4) + 200,
           int(prev.shape[1] / 4):int(prev.shape[1] / 4) + 200]
    # convert it into gray scaled
    prevgray = cv.cvtColor(prev, cv.COLOR_BGR2GRAY)
    inst = cv.DISOpticalFlow.create(cv.DISOPTICAL_FLOW_PRESET_MEDIUM)
    inst.setUseSpatialPropagation(False)

    use_temporal_propagation = True
    flow = None
    while True:
        _, img = cam.read()
        if img is None:
            break
        img = img[int(img.shape[0] / 4):int(img.shape[0] / 4) + 200,
              int(img.shape[1] / 4):int(img.shape[1] / 4) + 200]
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        if flow is not None and use_temporal_propagation:
            flow = inst.calc(prevgray, gray, warp_flow(flow, flow))
        else:
            flow = inst.calc(prevgray, gray, None)
        print(flow, flow.shape)
        prevgray = gray
    print('Done')


if __name__ == '__main__':
    main()
    cv.destroyAllWindows()
