#!/usr/bin/env python3
# =============================================================================
import numpy as np
import inspect
import math
import yaml
import copy
import sys
import cv2
import os

# =============================================================================
class Extrinsic:
    def __init__(self, ext_file_path, int_file_path):

        # ---------------------------------------------------------------------
        # Intrinsic Variables

        self._int_path = (
            None
            if int_file_path is None
            else os.path.dirname(os.path.abspath(int_file_path))
        )
        self._int_file = None if int_file_path is None else self._int_path
        self._intrinsic = (
            None
            if int_file_path is None
            else read_intrinsic_params(
                CONF_PATH=self._int_path, FILE_NAME=self._int_file
            )
        )

        # ---------------------------------------------------------------------
        self.M = None
        self.Mdst_pts = {"p1": None, "p2": None, "p3": None, "p4": None}
        self.Mpts = {"p1": None, "p2": None, "p3": None, "p4": None}
        self.pts = []
        self.src_width = None
        self.src_height = None
        self.src_scl = 1.0
        self.ppmx = None
        self.ppmy = None
        self.dst_warp_size = None
        self.M_warped_size = None
        self.intrinsic_used = False if self._intrinsic is None else True

        # Extrinsic variables
        self._ext_path = os.path.dirname(os.path.abspath(ext_file_path))
        self._ext_file = os.path.basename(ext_file_path)
        self.load_extrinsic()

    def load_extrinsic(self):
        """
            Load extrinsic calibration parameters from yaml file
            Args:
            returns:
        """

        file_path = os.path.join(self._ext_path, self._ext_file)
        if os.path.isfile(file_path):
            try:
                with open(file_path, "r") as stream:
                    data = yaml.safe_load(stream)

                # Cast variables to word with python variables
                self.M = np.array(data["M"])
                self.Mdst_pts = {
                    "p1": tuple(data["Mdst_pts"]["p1"]),
                    "p2": tuple(data["Mdst_pts"]["p2"]),
                    "p3": tuple(data["Mdst_pts"]["p3"]),
                    "p4": tuple(data["Mdst_pts"]["p4"]),
                }
                self.Mpts = {
                    "p1": tuple(data["Mpt_src"]["p1"]),
                    "p2": tuple(data["Mpt_src"]["p2"]),
                    "p3": tuple(data["Mpt_src"]["p3"]),
                    "p4": tuple(data["Mpt_src"]["p4"]),
                }
                self.pts = [tuple(pt) for pt in self.Mpts.values()]
                self.src_width = data["src_img_size"][0]
                self.src_height = data["src_img_size"][1]
                self.src_scl = data["img_scl"]
                self.ppmx = data["ppmx"]
                self.ppmy = data["ppmy"]
                self.dst_warp_size = tuple(data["dst_warp_size"])
                self.M_warped_size = tuple(data["M_warped_size"])
                self.intrinsic_used = data["intrinsic"]

                printlog(
                    msg="Extrinsic file {} loaded".format(self._ext_file),
                    msg_type="INFO",
                )
            except IOError as e:  # Report any error saving de data
                printlog(msg="Error loading extrinsic, {}".format(e), msg_type="ERROR")
        else:
            printlog(
                msg="No extrinsic file {} to load data".format(self._ext_file),
                msg_type="WARN",
            )

    def pt_src_to_dst(self, src_pt):

        if len(src_pt) == 2:
            src_pt = (src_pt[0], src_pt[1], 1)

        dst_pt = np.matmul(self.M, src_pt)
        dst_pt = dst_pt / dst_pt[2]

        return [int(dst_pt[0]), int(dst_pt[1])]

    def pt_dst_to_src(self, dst_pt):
        pass

    def get_warp_img(self, img_src):

        if img_src is None:
            return img_src

        img_src = cv2.resize(
            img_src,
            (self.src_width, self.src_height),
            interpolation=int(cv2.INTER_AREA),
        )

        return cv2.warpPerspective(dsize=self.dst_warp_size, src=img_src, M=self.M)


# =============================================================================
# OTHER UTILS - OTHER UTILS - OTHER UTILS - OTHER UTILS - OTHER UTILS - OTHER U
class bcolors:
    LOG = {
        "WARN": ["\033[33m", "WARN"],
        "ERROR": ["\033[91m", "ERROR"],
        "OKGREEN": ["\033[32m", "INFO"],
        "INFO": ["\033[0m", "INFO"],  # ['\033[94m', "INFO"],
        "BOLD": ["\033[1m", "INFO"],
        "GRAY": ["\033[90m", "INFO"],
        "USER": ["\033[95m", "INPUT"],
    }
    BOLD = "\033[1m"
    ENDC = "\033[0m"
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    GRAY = "\033[90m"
    UNDERLINE = "\033[4m"


def printlog(msg, msg_type="INFO", flush=True):
    org = os.path.splitext(os.path.basename(inspect.stack()[1][1]))[0].upper()
    caller = inspect.stack()[1][3].upper()
    _str = "[{}][{}][{}]: {}".format(bcolors.LOG[msg_type][1], org, caller, msg)
    print(bcolors.LOG[msg_type][0] + _str + bcolors.ENDC, flush=flush)


def read_intrinsic_params(CONF_PATH, FILE_NAME):
    """ 
        Loads intrinsic camera parameters from file  
    Args:
        file_path: `string` absolute path to yaml file
    Returns:
        file_path: `dict` intrinsic camera configuration
            dictionary
    """

    try:
        abs_path = os.path.join(CONF_PATH, FILE_NAME)

        if os.path.isfile(abs_path):
            with open(abs_path, "r") as stream:
                data_loaded = yaml.safe_load(stream)
        else:
            return None

        for key in [
            "camera_matrix",
            "distortion_coefficients",
            "rectification_matrix",
            "projection_matrix",
        ]:

            if key not in data_loaded:
                printlog(
                    msg="Intrinsic file {}, invalid".format(FILE_NAME), msg_type="ERROR"
                )
                return None

            data_loaded[key] = np.array(data_loaded[key]["data"]).reshape(
                data_loaded[key]["rows"], data_loaded[key]["cols"]
            )

        map1, map2 = cv2.initUndistortRectifyMap(
            cameraMatrix=data_loaded["camera_matrix"],
            distCoeffs=data_loaded["distortion_coefficients"],
            R=np.array([]),
            newCameraMatrix=data_loaded["camera_matrix"],
            size=(data_loaded["image_width"], data_loaded["image_height"]),
            m1type=cv2.CV_8UC1,
        )
        data_loaded["map1"] = map1
        data_loaded["map2"] = map2

    except Exception as e:
        printlog(
            msg="Loading intrinsic configuration file {}, {}".format(FILE_NAME, e),
            msg_type="ERROR",
        )
        return None

    printlog(
        msg="{} intrinsic configuration loaded".format(FILE_NAME), msg_type="OKGREEN"
    )

    return data_loaded


# =============================================================================
# IMAGE UTILS - IMAGE UTILS - IMAGE UTILS - IMAGE UTILS - IMAGE UTILS - IMAGE U


def dotline(src, p1, p2, color, thickness, Dl):
    """ 
        Draw in src image a doted line from p1 to p2 
    Args:
        p1: `tuple` (x, y) coordinate to start the line drawing
        p2: `tuple` (x, y) coordinate to end the line drawing
        color: `tuple` (R, G, B) color to draw the line
        thickness: `int` thickness of line
        Dl: `int` number of points in the line
    Returns:
        _: `tuple` input point undistorted
    """

    segments = discrete_contour((p1, p2), Dl)  # Discrete the line
    for segment in segments:  # Draw the discrete line with circles
        cv2.circle(src, segment, thickness, color, -1)

    return src


def discrete_contour(contour, Dl):
    """  Takes contour points to get a number of intermediate points
    Args:
        contour: List contour or list of points to get intermediate points
        Dl: int distance to get a point by segment
    Returns:
        new_contour: List new contour with intermediate points
    """

    # If contour has less of two points is not valid for operations
    if len(contour) < 2:
        print("Error: no valid segment")
        return contour

    # New contour variable
    new_contour = []

    # Iterate through all contour points
    for idx, cordinate in enumerate(contour):

        # Select next contour for operation
        if not idx == len(contour) - 1:
            next_cordinate = contour[idx + 1]
        else:
            next_cordinate = contour[0]

        # Calculate length of segment
        segment_lenth = math.sqrt(
            (next_cordinate[0] - cordinate[0]) ** 2
            + (next_cordinate[1] - cordinate[1]) ** 2
        )

        divitions = segment_lenth / Dl  # Number of new point for current segment
        dy = next_cordinate[1] - cordinate[1]  # Segment's height
        dx = next_cordinate[0] - cordinate[0]  # Segment's width

        if not divitions:
            ddy = 0  # Dy value to sum in Y axis
            ddx = 0  # Dx value to sum in X axis
        else:
            ddy = dy / divitions  # Dy value to sum in Y axis
            ddx = dx / divitions  # Dx value to sum in X axis

        # get new intermediate points in segments
        for idx in range(0, int(divitions)):
            new_contour.append(
                (int(cordinate[0] + (ddx * idx)), int(cordinate[1] + (ddy * idx)))
            )

    # Return new contour with intermediate points
    return new_contour


def overlay_image(l_img, s_img, pos, transparency):
    """ Overlay 's_img on' top of 'l_img' at the position specified by
        pos and blend using 'alpha_mask' and 'transparency'.
    Args:
        l_img: `cv2.mat` inferior image to overlay superior image
        s_img: `cv2.mat` superior image to overlay
        pos: `tuple`  position to overlay superior image [pix, pix]
        transparency: `float` transparency in overlayed image
    Returns:
        l_img: `cv2.mat` original image with s_img overlayed
    """

    # Get superior image dimensions
    s_img_height, s_img_width, s_img_channels = s_img.shape

    if s_img_channels == 3 and transparency != 1:
        s_img = cv2.cvtColor(s_img, cv2.COLOR_BGR2BGRA)
        s_img_channels = 4

    # Take 3rd channel of 'img_overlay' image to get shapes
    img_overlay = s_img[:, :, 0:4]

    # cords assignation to overlay image
    x, y = pos

    # Image ranges
    y1, y2 = max(0, y), min(l_img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(l_img.shape[1], x + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], l_img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], l_img.shape[1] - x)

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return l_img

    if s_img_channels == 4:
        # Get alphas channel
        alpha_mask = (s_img[:, :, 3] / 255.0) * transparency
        alpha_s = alpha_mask[y1o:y2o, x1o:x2o]
        alpha_l = 1.0 - alpha_s

        # Do the overlay with alpha channel
        for c in range(0, l_img.shape[2]):
            l_img[y1:y2, x1:x2, c] = (
                alpha_s * img_overlay[y1o:y2o, x1o:x2o, c]
                + alpha_l * l_img[y1:y2, x1:x2, c]
            )

    elif s_img_channels < 4:
        # Do the overlay with no alpha channel
        if l_img.shape[2] == s_img.shape[2]:
            l_img[y1:y2, x1:x2] = s_img[y1o:y2o, x1o:x2o]
        else:
            printlog(
                msg="Error: to overlay images should have the same color channels",
                msg_type="ERROR",
            )
            return l_img

    # Return results
    return l_img


def print_text_list(
    img, tex_list, color=(0, 0, 255), orig=(10, 25), fontScale=0.7, y_jump=25
):
    """
        Print a text list on image in desending order
        Args:
            img: 'cv2.math' image to draw components
            tex_list: 'list' list with text to print/draw
            color: 'list' bgr opencv color of text
            orig: 'tuple' origin to start draw components
            fontScale: 'float' text font scale
            y_jump: 'int' jump or space between lines
        returns:
    """

    for idx, text in enumerate(tex_list):
        cv2.putText(
            img=img,
            text=text,
            org=(orig[0], int(orig[1] + y_jump * idx)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=fontScale,
            color=(0, 0, 0),
            thickness=3,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            img=img,
            text=text,
            org=(orig[0], int(orig[1] + y_jump * idx)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=fontScale,
            color=color,
            thickness=1,
            lineType=cv2.LINE_AA,
        )


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    """
        Scale and image keeping the proportion of it, for example if width is
        None but height setted tiy get an image of height size but keeping the 
        width proportion. If width and height both are setted you get an image
        keeping the original aspect ration centered in the new image.
        Args:
            img: 'cv2.math' image to resize
            width: 'int' new width size to resize
            height: 'int' new height size to resize
        returns:
    """

    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # resize the image
    if width is not None and height is not None:
        background = np.zeros((height, width, 3), np.uint8)
        y_pos = int((height * 0.5) - (resized.shape[0] * 0.5))
        background[y_pos : y_pos + resized.shape[0], 0 : resized.shape[1]] = resized
        return background

    # return the resized image
    return resized


def create_radar(size=300, div=30, color=(0, 0, 255), img=None, thickness=1, Dl=6):
    """
        Create radar image
        Args:
            size: 'cv2.math' image to resize
            div: 'int' horizontal and vertical divitions
            color: 'int' lines BGR color 
            img: 'cv2.math' background image 
            thickness: 'int' radar lines thickness
        returns:
            radar_img: 'cv2.math' radar image
    """

    # Create radar base image
    if img is not None:
        radar_img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    else:
        radar_img = np.zeros((size, size, 3), dtype=np.uint8)

    # Draw first elements
    cv2.rectangle(
        radar_img,
        (0, 0),
        (radar_img.shape[1] - 1, radar_img.shape[0] - 1),
        color,
        thickness + 1,
    )

    # Draw vertical lines
    for i in range(0, radar_img.shape[1], int(radar_img.shape[1] / (div + 1))):
        cv2.line(
            img=radar_img,
            pt1=(i, 0),
            pt2=(i, radar_img.shape[0]),
            color=color,
            thickness=thickness,
        )
        # dotline(src=radar_img, p1=(i , 0), p2=(i, radar_img.shape[0]),
        #     color=(0, 255, 0), thickness=thickness, Dl=Dl)
    for i in range(0, radar_img.shape[0], int(radar_img.shape[0] / (div + 1))):
        cv2.line(
            img=radar_img,
            pt1=(0, i),
            pt2=(radar_img.shape[1], i),
            color=color,
            thickness=thickness,
        )
        # dotline(src=radar_img, p1=(0 , i), p2=(radar_img.shape[1], i),
        #     color=(0, 255, 0), thickness=thickness, Dl=Dl)

    return radar_img


# =============================================================================
# GEOMETRY/MATH OPS - GEOMETRY/MATH OPS - GEOMETRY/MATH OPS - GEOMETRY/MATH OPS


def closest_neighbor(pts_list, x, y):
    """
        Description
        Args:
            pts_list: 'list' list of points in XY coordinates
            x: 'int' X axis of coordinate to look for closets neighbor
            y: 'int' Y axis of coordinate to look for closets neighbor
        returns:
            if type(pts_list) == list:
                _: 'int' index of the closets neighbor
                _: 'tuple' point in XY coordinates of the closets neighbor
            if type(pts_list) == dict:
                _: 'int' key of the closets neighbor
                _: 'tuple' point in XY coordinates of the closets neighbor  
    """

    if type(pts_list) == list:
        dist_list = [
            (i, np.sqrt((pts_list[i][0] - x) ** 2 + (pts_list[i][1] - y) ** 2))
            for i in range(0, 4)
        ]
        dist_list.sort(key=sort_pt)
        idx = dist_list[0][0]
        return idx, pts_list[idx]

    elif type(pts_list) == dict:
        clt_key = None
        clt_d = float("inf")
        for key, pt in pts_list.items():
            if pt is None:
                continue
            d = np.sqrt((pt[0] - x) ** 2 + (pt[1] - y) ** 2)
            if d < clt_d:
                clt_key = key
                clt_d = d
        if clt_key == None:
            return None, None
        else:
            return clt_key, pts_list[clt_key]


def sort_pt(point):
    return point[1]


def get_projection_point_dst(coords_src, M):
    """ 
        this function returns the cords of a point in "X" and "Y" in a surface 
        according to the Rotation Matrix surface and origin point in an original 
        source image
    Args:
        coords_src: `tuple` cords in the original image
        M: `numpy.narray` rotation matrix from geometric projection to original 
    Returns:
        _: return the projected point according with rotation matrix "M" and 
        the original point 'coords_src'
    """

    if len(coords_src) == 2:
        coords_src = (coords_src[0], coords_src[1], 1)
    np.array(coords_src)

    # ------------------------------------------------------------
    coords_dst = np.matmul(M, coords_src)
    coords_dst = coords_dst / coords_dst[2]

    # ------------------------------------------------------------
    # return results
    return (int(coords_dst[0]), int(coords_dst[1]))


def get_projection_point_src(coords_dst, INVM):
    """ 
        Get the cords of a point in 'X' and 'Y' from projected area to source 
        area in original image
    Args:
        coords_src: `tuple` cords in the original image
        INVM: `numpy.narray` inverse of rotation matrix from geometric projection to original 
    Returns:
        _: return the projected point according with rotation matrix 'M' and 
            the original point 'coords_dst'  
    """

    # ------------------------------------------------------------
    coords_src = np.matmul(INVM, coords_dst)
    coords_src = coords_src / coords_src[2]

    # ------------------------------------------------------------
    # return results
    return (int(coords_src[0]), int(coords_src[1]))


def get_rot_matrix(p1, p2, p3, p4, UNWARPED_SIZE):
    """     
        Calculates rotation matrix from points
    Args:
        p1: `tuple` First point of Coordinates of quadrangle vertices in the source image
        p2: `tuple` Second point of Coordinates of quadrangle vertices in the source image
        p3: `tuple` Third point of Coordinates of quadrangle vertices in the source image
        p4: `tuple` Fourth point of Coordinates of quadrangle vertices in the source image
        UNWARPED_SIZE: `tuple` surface projection space size
    Returns:
        M: `numpy.darray` The matrix for a perspective transform
    """

    # Calculate rotation matrix from surface from original source image to projected four points surfaces
    src_points = np.array([p1, p2, p3, p4], dtype=np.float32)
    dst_points = np.array(
        [
            [0, 0],  # p1
            [UNWARPED_SIZE[0], 0],  # p2
            [UNWARPED_SIZE[0], UNWARPED_SIZE[1]],  # p3
            [0, UNWARPED_SIZE[1]],
        ],  # p4
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(src_points, dst_points)

    return M


# =============================================================================
# OBJECT DETECTION UTILS - OBJECT DETECTION UTILS - OBJECT DETECTION UTILS - OB


def draw_predictions(img_src, predictions, normalized=False):
    """     
        Draw bounding boxes predictions
    Args:
        img_src: `cv2.math` image to draw components
        predictions: `list` list with object detection predictions
        normalized: `boolean` Enable/Disable predictions box coordinated normalized
    Returns:
        img_src: `cv2.math` image with components to drawn
    """

    if len(predictions):
        for pred in predictions:
            if str(pred["name"]) != "person":
                continue

            x, y, x2, y2 = pred["box"]
            if normalized:
                x = int(x * img_src.shape[1])
                y = int(y * img_src.shape[0])
                x2 = int(x2 * img_src.shape[1])
                y2 = int(y2 * img_src.shape[0])

            w = x2 - x
            h = y2 - y
            cen = [int(x + w / 2), int(y + h / 2)]

            color = (0, 0, 255)
            if pred["safe"]:
                color = (0, 255, 0)
            if not pred["in_cnt"]:
                color = (150, 150, 150)

            cv2.circle(img_src, tuple(cen), 2, (255, 255, 255), -1)
            cv2.rectangle(
                img_src,
                (x, y),
                (x2, y2),
                color,
                2,
            )

            cv2.putText(
                img=img_src,
                text=str(round(pred["confidence"], 2)),
                org=(x, y - 7),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.3,
                color=(0, 0, 0),
                thickness=2,
                lineType=cv2.LINE_AA,
            )
            cv2.putText(
                img=img_src,
                text=str(round(pred["confidence"], 2)),
                org=(x, y - 7),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.3,
                color=(255, 255, 255),
                thickness=1,
                lineType=cv2.LINE_AA,
            )

            cen = [int(x + w / 2), int(y + h)]
            cv2.circle(img_src, tuple(cen), 2, (0, 255, 255), -1)

    return img_src


def norm_predictions(predictions, img_shape):
    """     
        Normalize predictions coordinates
    Args:
        predictions: `list` list with object detection predictions
        img_shape: `tuple` image shape
    Returns:
        nom_predic: `list` list with object detection predictions normalized
    """

    if predictions is None:
        return None

    nom_predic = copy.deepcopy(predictions)

    for idx, pred in enumerate(nom_predic):

        x, y, x2, y2 = pred["box"]

        nom_predic[idx]["box"] = (
            float(x / img_shape[1]),
            float(y / img_shape[0]),
            float(x2 / img_shape[1]),
            float(y2 / img_shape[0]),
        )

        if "box_base" in pred.keys():
            x, y = pred["box_base"]
            nom_predic[idx]["box_base"] = [
                float(x / img_shape[1]),
                float(y / img_shape[0]),
            ]

    return nom_predic


def get_base_predictions(predictions):
    """     
        Add rectangle inferior base of bounding boxes predictions and other 
        detection features
    Args:
        predictions: `list` list with object detection predictions
    Returns:
    """

    if len(predictions):
        for idx, pred in enumerate(predictions):

            x, y, x2, y2 = pred["box"]
            w = x2 - x
            h = y2 - y

            cen = [float(x + w / 2.0), float(y + h)]
            pred["box_center"] = cen
            pred["idx"] = idx
            pred["safe"] = True
            pred["neighbors"] = []
            pred["box_base_src"] = [cen[0], y2]
            pred["box_base_dst"] = None
            pred["box_base_dst_norm"] = None
            pred["in_cnt"] = True

    return predictions


# =============================================================================
