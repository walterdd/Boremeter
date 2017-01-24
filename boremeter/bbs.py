import cv2


def are_close(bb1, bb2):
    
    # check if two bounding boxes are close and have similar shapes
    
    if abs(bb1[2]*bb1[3] - bb2[2]*bb2[3]) > max(bb1[2]*bb1[3], bb2[2]*bb2[3]) / 4:
        return False
    
    return (abs(bb1[0] + bb1[2] / 2 - bb2[0] - bb2[2] / 2) < bb1[2] / 2) and \
            (abs(bb1[1] + bb1[3] / 2 - bb2[1] - bb2[3] / 2) <  bb1[3] / 2)


def scale_bb(x, y, w, h, scale):
    
    # returns scaled parameters of the bounding box
    
    return [
            int(x - w * (scale - 1) / 2), 
            int(y - h * (scale - 1) / 2), 
            int(w * (scale)),
            int(h * (scale))
            ]

def intersection_area(x1, y1, w1, h1, x2, y2, w2, h2):
    x11 = x1
    x21 = x2
    x12 = x1 + w1
    x22 = x2 + w2
    y11 = y1
    y21 = y2
    y12 = y1 + h1
    y22 = y2 + h2
    x_overlap = max(0, min(x12,x22) - max(x11,x21));
    y_overlap = max(0, min(y12,y22) - max(y11,y21));
    overlapArea = x_overlap * y_overlap;
    return overlapArea

def square(bb):
    
    return bb[2] * bb[3]

def intersection_area_by_bbs(bb1, bb2):
    
    # count square of intersection
    
    x_left_bound = sorted([bb1[0], bb2[0]])
    x_right_bound = sorted([bb1[0] + bb1[2], bb2[0] + bb2[2]])
    y_lower_bound = sorted([bb1[1], bb2[1]])
    y_upper_bound = sorted([bb1[1] + bb1[3], bb2[1] + bb2[3]])
    return (x_right_bound[0] - x_left_bound[1]) * (y_upper_bound[0] - y_lower_bound[1])

def have_intersection(bb1, bb2):
    
    # check if two bounding boxes have an intersection

    return not (bb1[0] + bb1[2] < bb2[0] \
                or bb2[0] + bb2[2] < bb1[0] \
                or bb1[1] + bb1[3] < bb2[1] \
                or bb2[1] + bb2[3] < bb1[1])


def pts_to_bb(pts):
    x = min(pts[0][0], pts[1][0], pts[2][0], pts[3][0]) 
    y = min(pts[0][1], pts[1][1], pts[2][1], pts[3][1]) 
    w = max(pts[0][0], pts[1][0], pts[2][0], pts[3][0]) - x
    h = max(pts[0][1], pts[1][1], pts[2][1], pts[3][1]) - y
    return [x, y, w, h]
