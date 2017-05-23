import numpy as np
import cv2
import math
from tqdm import tqdm

class headPoseEstimator:
    def __init__(self, frame_shape, method=cv2.SOLVEPNP_DLS, head_3d_model=None, distortion=np.zeros((4,1))):
        if head_3d_model:
            self.model_points = head_3d_model
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ])
        focal_length = frame_shape[1]
        center = (frame_shape[1]/2, frame_shape[0]/2)
        self.camera_matrix = np.array(
                         [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = "double"
                         )
        self.dist_coeffs = distortion
        self.method=method

    def points2landmarks(self, points):
        left_eye_corner = [points[0], points[5]]
        right_eye_corner = [points[1], points[6]]
        nose_tip = [points[2], points[7]]
        left_mouth_corner = [points[3], points[8]]
        right_mouth_corner = [points[4], points[9]]

        landmarks = np.array([nose_tip,
                                 left_eye_corner,
                                 right_eye_corner,
                                 left_mouth_corner,
                                 right_mouth_corner], dtype="double")
        return landmarks

    def get_pose(self, landmarks):
        (success, rotation_vector, translation_vector) = cv2.solvePnP(
            self.model_points.reshape((5,1,3)), landmarks.reshape((5,1,2)), self.camera_matrix, self.dist_coeffs, flags=self.method)
        if not success:
            raise RuntimeError("cv2.solvePnP did not succeed")
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        raw_roll, raw_pitch, raw_yaw = mat2euler(np.linalg.inv(rotation_matrix))

        pitch = raw_yaw - np.pi
        yaw = raw_pitch
        roll = raw_roll

        roll = round(np.rad2deg(roll), 4)
        pitch = round(np.rad2deg(pitch), 4)
        yaw = round(np.rad2deg(yaw), 4)
        if pitch < -300:
            pitch += 360
        return rotation_vector, translation_vector, yaw, pitch, roll

    def get_all_poses(self, points_list):
        poses = [self.get_pose(self.points2landmarks(points)) for points in points_list]
        return poses

    def get_all_YPR(self, points_list):
        angles = np.array([self.get_pose(self.points2landmarks(points))[2:] for points in points_list])
        return angles

    def get_all_directions(self, points_list):
        poses = self.get_all_poses(points_list)
        end_points2D = [cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), poses[i][0],
                                          poses[i][1], self.camera_matrix, self.dist_coeffs)[0]
                        for i in range(len(poses))]
        angles = np.array([poses[i][2:] for i in range(len(poses))])
        return angles, end_points2D

def mat2euler(M, cy_thresh=None):
    ''' Discover Euler angle vector from 3x3 matrix

    Parameters
    ----------
    M : array-like, shape (3,3)
    cy_thresh : None or scalar, optional
       threshold below which to give up on straightforward arctan for
       estimating x rotation.  If None (default), estimate from
       precision of input.

    Returns
    -------
    z : scalar
    y : scalar
    x : scalar
       Rotations in radians around z, y, x axes, respectively
    '''
    M = np.asarray(M)
    if cy_thresh is None:
        try:
            cy_thresh = np.finfo(M.dtype).eps * 4
        except ValueError:
            cy_thresh = _FLOAT_EPS_4
    r11, r12, r13, r21, r22, r23, r31, r32, r33 = M.flat
    # cy: sqrt((cos(y)*cos(z))**2 + (cos(x)*cos(y))**2)
    cy = math.sqrt(r33*r33 + r23*r23)
    if cy > cy_thresh: # cos(y) not close to zero, standard form
        z = math.atan2(-r12,  r11) # atan2(cos(y)*sin(z), cos(y)*cos(z))
        y = math.atan2(r13,  cy) # atan2(sin(y), cy)
        x = math.atan2(-r23, r33) # atan2(cos(y)*sin(x), cos(x)*cos(y))
    else: # cos(y) (close to) zero, so x -> 0.0 (see above)
        # so r21 -> sin(z), r22 -> cos(z) and
        z = math.atan2(r21,  r22)
        y = math.atan2(r13,  cy) # atan2(sin(y), cy)
        x = 0.0
    return z, y, x

def estimate_poses(detected_faces, frame_shape, detection_step):

    hpe = headPoseEstimator(frame_shape)

    detected_faces.loc[:, 'yaw'] = np.zeros(detected_faces.shape[0])
    detected_faces.loc[:, 'pitch'] = np.zeros(detected_faces.shape[0])
    detected_faces.loc[:, 'roll'] = np.zeros(detected_faces.shape[0])
    detected_faces.loc[:, 'rotation_vector'] = np.zeros(detected_faces.shape[0])
    detected_faces.loc[:, 'translation_vector'] = np.zeros(detected_faces.shape[0])

    for i, face_row in tqdm(detected_faces.iterrows(), total=detected_faces.shape[0]):
        if face_row['frame'] % detection_step == 0:
            points = face_row['points'].split(',')
            landmarks = hpe.points2landmarks(points)
            rotation_vector, translation_vector, yaw, pitch, roll = hpe.get_pose(landmarks)
            detected_faces.loc[i, 'yaw'] = yaw
            detected_faces.loc[i, 'pitch'] = pitch
            detected_faces.loc[i, 'roll'] = roll
            detected_faces.loc[i, 'rotation_vector'] = ','.join([str(r[0]) for r in rotation_vector])
            detected_faces.loc[i, 'translation_vector'] = ','.join([str(t[0]) for t in translation_vector])

    return detected_faces















