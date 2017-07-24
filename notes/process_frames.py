import numpy as np
from lesson_functions import *

def process_frames(img):
    rectangles = []
    colorspace = 'YUV' # It can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 11
    pix_per_cell = 16
    cell_per_block = 2
    hog_channel = 'ALL' # It can be 0,1,2,'ALL'

    ystart,ystop,scale = 400,464,1.0
    detected_rects = find_cars(img,ystart,ystop,scale,colorspace,hog_channel,svc,None,orient,pix_per_cell,cell_per_block,None,None,show_all_rectangles=True)
    rectangles.append(detected_rects)

    ystart,ystop,scale = 416,480,1.0
    detected_rects = find_cars(img,ystart,ystop,scale,colorspace,hog_channel,svc,None,orient,pix_per_cell,cell_per_block,None,None,show_all_rectangles=True)
    rectangles.append(detected_rects)

    ystart,ystop,scale = 400,496,1.5
    detected_rects=find_cars(test_img, ystart, ystop, scale, colorspace, hog_channel, svc, None,
                       orient, pix_per_cell, cell_per_block, None, None)
    rectangles.append(detected_rects)

    ystart,ystop,scale = 432,528,1.5
    detected_rects = find_cars(test_img, ystart, ystop, scale, colorspace, hog_channel, svc, None,
                           orient, pix_per_cell, cell_per_block, None, None)
    rectangles.append(detected_rects)

    ystart,ystop,scale = 400,528,2.0
    ystop = 528
    scale = 2.0
    detected_rects = find_cars(test_img, ystart, ystop, scale, colorspace, hog_channel, svc, None,
                           orient, pix_per_cell, cell_per_block, None, None)
    rectangles.append(detected_rects)

    ystart,ystop,scale = 432,560,2.0
    detected_rects = find_cars(test_img, ystart, ystop, scale, colorspace, hog_channel, svc, None,
                           orient, pix_per_cell, cell_per_block, None, None)
    rectangles.append(detected_rects)

    ystart,ystop,scale = 400,596,3.5
    detected_rects = find_cars(test_img, ystart, ystop, scale, colorspace, hog_channel, svc, None,
                           orient, pix_per_cell, cell_per_block, None, None)
    rectangles.append(detected_rects)

    ystart,ystop,scale = 464,560,3.5
    detected_rects = find_cars(test_img, ystart, ystop, scale, colorspace, hog_channel, svc, None,
                           orient, pix_per_cell, cell_per_block, None, None)
    rectangles.append(detected_rects)

    rectangles = [item for sublist in rectangles for item in sublist]

    heatmap_img = np.zeros_like(img[:,:,0])
    heatmap_img = add_heat(heatmap_img, rectangles)
    heatmap_img = apply_threshold(heatmap_img, 1)
    labels = label(heatmap_img)
    draw_img, rects = draw_labeled_bboxes(np.copy(img), labels)
    return draw_img



