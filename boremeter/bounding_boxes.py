
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
        self.x -= int(self.w * (scale - 1) / 2)
        self.y -= int(self.h * (scale - 1) / 2)
        self.x = max(self.x, 0)
        self.y = max(self.y, 0)
        self.w = int(self.w * scale)
        self.h = int(self.h * scale)

    def get_area(self):
        return self.w * self.h

    def get_center(self):
        return self.x + self.w / 2, self.y + self.h / 2

    def copy(self):
        return BoundingBox(self.x, self.y, self.w, self.h)


def are_close(bb1, bb2):
    
    # check if two bounding boxes are close and have similar shapes
    
    if abs(bb1.get_area() - bb2.get_area()) > max(bb1.get_area(), bb2.get_area()) / 4:  # check areas
        return False
    
    return ((abs(bb1.get_center()[0] - bb2.get_center()[0]) < bb1.w / 2) and
            (abs(bb1.get_center()[1] - bb2.get_center()[1]) < bb1.h / 2))


def intersection_area(bb1, bb2):
    x_overlap = max(0, min(bb1.right, bb2.right) - max(bb1.left, bb2.left))
    y_overlap = max(0, min(bb1.bottom, bb2.bottom) - max(bb1.top, bb2.top))

    overlap_area = x_overlap * y_overlap

    return overlap_area


def have_intersection(bb1, bb2):
    
    # check if two bounding boxes have an intersection

    return intersection_area(bb1, bb2) > 0


def pts_to_bb(pts):

    # converts coordinates or corners to Bb

    x = min(pts[0][0], pts[1][0], pts[2][0], pts[3][0]) 
    y = min(pts[0][1], pts[1][1], pts[2][1], pts[3][1]) 
    w = max(pts[0][0], pts[1][0], pts[2][0], pts[3][0]) - x
    h = max(pts[0][1], pts[1][1], pts[2][1], pts[3][1]) - y
    return BoundingBox(x, y, w, h)
