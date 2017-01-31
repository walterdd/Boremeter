
class BoundingBox:
    """
    Bb - class for bounding boxes

          x (left)    right
              |          |
              |          |
    y (top)___|__________|
              |          |
              | BOUNDING |
              |   BOX    | h
    bottom____|__________|
                   w

    """
    def __init__(self, x, y, w, h):
        self.x, self.y, self.w, self.h = x, y, w, h

    @property
    def left(self):
        return self.x

    @property
    def right(self):
        return self.x + self.w

    @property
    def top(self):
        return self.y

    @property
    def bottom(self):
        return self.y + self.h

    def get(self):
        return self.x, self.y, self.w, self.h

    def resize(self, scale):
        new_x = self.x - int(self.w * (scale - 1) / 2)
        new_y = self.y - int(self.h * (scale - 1) / 2)
        new_x = max(new_x, 0)
        new_y = max(new_y, 0)
        new_w = int(self.w * scale)
        new_h = int(self.h * scale)
        return BoundingBox(new_x, new_y, new_w, new_h)

    @property
    def area(self):
        return self.w * self.h

    @property
    def center(self):
        return self.x + self.w / 2, self.y + self.h / 2

    def copy(self):
        return BoundingBox(self.x, self.y, self.w, self.h)


def bboxes_are_close(bb1, bb2):
    """
    Check if two bounding boxes are close and have similar shapes.
    """
    
    if abs(bb1.area - bb2.area) > max(bb1.area, bb2.area) / 4:  # check areas
        return False

    center1_x, center1_y = bb1.center
    center2_x, center2_y = bb2.center
    return ((abs(center1_x - center2_x) < bb1.w / 2) and
            (abs(center1_y - center2_y) < bb1.h / 2))


def intersection_area(bb1, bb2):
    x_overlap = max(0, min(bb1.right, bb2.right) - max(bb1.left, bb2.left))
    y_overlap = max(0, min(bb1.bottom, bb2.bottom) - max(bb1.top, bb2.top))

    overlap_area = x_overlap * y_overlap

    return overlap_area


def have_intersection(bb1, bb2):
    """
    Check if two bounding boxes have an intersection.
    """
    return intersection_area(bb1, bb2) > 0


def pts_to_bb(pts):

    """
    Converts coordinates or corners to a BoundingBox object.
    """

    x = min(pts[0][0], pts[1][0], pts[2][0], pts[3][0]) 
    y = min(pts[0][1], pts[1][1], pts[2][1], pts[3][1]) 
    w = max(pts[0][0], pts[1][0], pts[2][0], pts[3][0]) - x
    h = max(pts[0][1], pts[1][1], pts[2][1], pts[3][1]) - y
    return BoundingBox(x, y, w, h)
