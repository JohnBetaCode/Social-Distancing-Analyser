#!/usr/bin/env python3
# =============================================================================
import sys

# For user with ros1 installed in their system
cv2ros_path = "/opt/ros/kinetic/lib/python2.7/dist-packages"
if cv2ros_path in sys.path:
    sys.path.remove(cv2ros_path)

import numpy as np
import getopt
import yaml
import math
import glob
import cv2
import os

sys.path.append(".")

from python_utils.vision_utils import get_projection_point_dst
from python_utils.vision_utils import read_intrinsic_params
from python_utils.vision_utils import closest_neighbor
from python_utils.vision_utils import get_rot_matrix
from python_utils.vision_utils import image_resize
from python_utils.vision_utils import printlog
from python_utils.vision_utils import dotline

# =============================================================================
class calibrator:
    def __init__(self, img, img_scl=1.0, intrinsic=None, extrinsic=None):
        """
            Class constructor for extrinsic calibration objects
            Args:
                img: 'cv2.math' image to calibrate extrinsic
                img_scl: 'float' scale factor to scale input image (resize)
                    set 1.0 for less than 640x360 images
                intrinsic: 'dict' dictionary with intrinsic calibration
                    see *.yaml files for more variables descriptions
            returns:
        """

        # Set extrinsic file and path
        self._ext_file = "extrinsic.yaml" if extrinsic is None else extrinsic
        fwd = os.path.dirname(os.path.abspath(__file__))
        fwd = os.path.abspath(os.path.join(fwd, os.pardir))
        self._ext_path = os.path.join(fwd, "configs")

        # Load menu/help/options image
        self._img_menu = cv2.imread(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "figures",
                "extrinsic_menu.jpg",
            )
        )
        self._show_menu = True
        self._in_gerion = True

        # Extrinsic environment variables
        self._VISION_CAL_WARPED_WIDTH = int(
            os.getenv(key="VISION_CAL_WARPED_WIDTH", default=300)
        )
        self._VISION_CAL_WARPED_HEIGHT = int(
            os.getenv(key="VISION_CAL_WARPED_HEIGHT", default=300)
        )
        self._VISION_CAL_WARPED_SIZE = (
            self._VISION_CAL_WARPED_WIDTH,
            self._VISION_CAL_WARPED_HEIGHT,
        )
        self._WIN_WARP_HOZ_DIV = int(os.getenv(key="WIN_WARP_HOZ_DIV", default=25))
        self._WIN_WARP_VER_DIV = int(os.getenv(key="WIN_WARP_VER_DIV", default=25))

        # EXtrinsic and intrinsic variables
        self._intrinsic = intrinsic
        self._M = None
        self._Mpts = {"p1": None, "p2": None, "p3": None, "p4": None}
        self._Mdst_pts = {"p1": None, "p2": None, "p3": None, "p4": None}
        self._MOriginpts = []
        self.ppmx = None
        self.ppmy = None
        self.warp_width = None
        self.warp_height = None

        # Image source
        self.img_src = cv2.resize(
            src=img,
            dsize=(int(img.shape[1] * img_scl), int(img.shape[0] * img_scl)),
            interpolation=int(cv2.INTER_AREA),
        )

        # Perform intrinsic if exists
        if self._intrinsic is not None:

            # Resize image if size different to intrinsic calibration size
            if (
                self.img_src.shape[1] != self._intrinsic["image_width"]
                and self.img_src.shape[0] != self._intrinsic["image_height"]
            ):
                self.img_src = cv2.resize(
                    self.img_src,
                    (self._intrinsic["image_width"], self._intrinsic["image_height"]),
                    interpolation=int(cv2.INTER_AREA),
                )

            # Undistord image if intrinsic calibration
            self.img_src = cv2.remap(
                self.img_src,
                self._intrinsic["map1"],
                self._intrinsic["map2"],
                cv2.INTER_LINEAR,
            )

        self.img_width = self.img_src.shape[1]
        self.img_height = self.img_src.shape[0]

        # Other process variables
        self.img_scl = img_scl
        self.img_gui = self.img_src.copy()
        self.img_sky = None

        # src Window variables
        self._win_name = "extrinsic_calibrator"
        self._win_event = False
        self._win_time = 50
        cv2.namedWindow(self._win_name)
        cv2.setMouseCallback(self._win_name, self._win_mouse_event)
        # Mouse event variables
        self.pts = []
        self.mouse_current_pos = [0, 0]
        self.pts_idx = 0

        # Birdview window
        self._win_bird_name = "bird_view"
        self._win_bird_event = False
        self._win_bird_time = 50
        self._win_bird_max_width = 500.0
        self._win_bird_max_height = 500.0
        self._win_bird_width = 1.0
        self._win_bird_height = 1.0
        cv2.namedWindow(self._win_bird_name)
        cv2.setMouseCallback(self._win_bird_name, self._win_bird_mouse_event)
        # Mouse event variables
        self.bird_pts = []
        self.bird_mouse_current_pos = [0, 0]

        # Load data
        self._load_extrinsic()

    def _win_mouse_event(self, event, x, y, flags, param):
        """
            Callback for mouse events in the extrinsic calibrator window 
            Args:
                event: 'int' mouse event
                x: 'int' mouse event x axis position 
                y: 'int' mouse event y axis position 
                flags: 'list[int]' flags given by mouse event 
                param: 'XXXX' XXXX
            returns:
        """

        # Do nothing if in menu
        if self._show_menu:
            return

        self._win_event = True
        self.mouse_current_pos = [x, y]

        # Add point
        if event == cv2.EVENT_LBUTTONDOWN:

            # If already assigned all points for a surface matrix projection
            if len(self.pts) == 4:

                # Change geometry by displacement
                idx, _ = closest_neighbor(self.pts, x, y)
                self.pts[idx] = (x, y)
                self.pts_idx = idx

                # Re-assign surface matrix projection points
                key, _ = closest_neighbor(self._Mpts, x, y)
                if key is not None:
                    self._Mpts[key] = (x, y)

                # Update window elements of bird view window
                self._win_bird_event = True
                self._calibrate_extrinsic()

                return

            # Add point to user's points list
            else:
                self.pts.append((x, y))
                self.pts_idx = len(self.pts) - 1
                self._assing_mpt(pt_idx=len(self.pts))

                if len(self.pts) == 4:
                    # Update window elements of bird view window
                    self._win_bird_event = True
                    self._calibrate_extrinsic()

                return

    def _win_bird_mouse_event(self, event, x, y, flags, param):
        """
            Callback for mouse events in the bird/sky view window 
            Args:
                event: 'int' mouse event
                x: 'int' mouse event x axis position 
                y: 'int' mouse event y axis position 
                flags: 'list[int]' flags given by mouse event 
                param: 'XXXX' XXXX
            returns:
        """

        self._win_bird_event = True

        if event == cv2.EVENT_RBUTTONDOWN:
            self.bird_pts = []
            self.bird_mouse_current_pos = [0, 0]
            return

        # Update bird/sky view window components
        if self.img_sky is not None:
            x = int((x / self.img_sky.shape[1]) * self._win_bird_width)
            y = int((y / self.img_sky.shape[0]) * self._win_bird_height)

        # Re-scale x and y axis to fit original image size
        self.bird_mouse_current_pos = [x, y]

        if len(self.bird_pts) == 1:
            dv = abs(self.bird_pts[0][1] - y)
            dh = abs(self.bird_pts[0][0] - x)
            if dv >= dh:
                self.bird_mouse_current_pos[0] = self.bird_pts[0][0]
            else:
                self.bird_mouse_current_pos[1] = self.bird_pts[0][1]

        # Add point
        if event == cv2.EVENT_LBUTTONDOWN:

            self.bird_pts.append(
                (self.bird_mouse_current_pos[0], self.bird_mouse_current_pos[1])
            )

            if len(self.bird_pts) == 2:
                dp = int(math.sqrt((dv) ** 2 + (dh) ** 2))
                printlog(
                    msg="Please set distance in meters [m] for {} {}"
                    " pixels".format(dp, "vertical" if dv >= dh else "horizontal"),
                    msg_type="USER",
                )
                dm = input("input [m]: ")
                self._update_pix_relation(
                    dm=float(dm), dp=int(dp), relation="dv" if dv >= dh else "dh"
                )
                self.bird_pts = []

    def _draw_measure(self, img):
        """
            Draw user's measure over sky/bird image window
            Args:
                img: 'cv2.math' image to perform operations and draw elements
            returns:
        """

        if len(self.bird_pts) == 1:
            cv2.line(
                img=img,
                pt1=self.bird_pts[0],
                pt2=tuple(self.bird_mouse_current_pos),
                color=(255, 0, 255),
                thickness=2,
            )

        elif len(self.bird_pts) == 2:
            cv2.line(
                img=img,
                pt1=self.bird_pts[0],
                pt2=self.bird_pts[1],
                color=(255, 0, 0),
                thickness=2,
            )

    def _draw_points(self, img):
        """
            Draw user's points in the src image space, and the points of the 
                matrix surfarce projection
            Args:
                img: 'cv2.math' image to perform operations and draw elements
            returns:
        """

        # Draw surface lines
        if self._Mpts["p1"] is not None and self._Mpts["p3"] is not None:
            dotline(
                src=img,
                p1=self._Mpts["p1"],
                p2=self._Mpts["p3"],
                color=(255, 255, 255),
                thickness=1,
                Dl=5,
            )
        if self._Mpts["p2"] is not None and self._Mpts["p4"] is not None:
            dotline(
                src=img,
                p1=self._Mpts["p2"],
                p2=self._Mpts["p4"],
                color=(255, 255, 255),
                thickness=1,
                Dl=5,
            )

        # Draw user's points
        for pt in self.pts:  # Inner points
            cv2.circle(img=img, center=pt, radius=2, color=(0, 0, 255), thickness=-1)
        if len(self.pts):  # Draw current index point
            cv2.circle(
                img=img,
                center=self.pts[self.pts_idx],
                radius=10,
                color=(0, 255, 255),
                thickness=3,
            )

        # Draw matrix surfarce projection points
        for key, pt in self._Mpts.items():
            if pt is not None:
                cv2.putText(
                    img=img,
                    text="{}".format(key),
                    org=(pt[0] - 8, pt[1] - 8),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.7,
                    color=(0, 0, 0),
                    thickness=3,
                    lineType=cv2.LINE_AA,
                )
                cv2.putText(
                    img=img,
                    text="{}".format(key),
                    org=(pt[0] - 8, pt[1] - 8),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.7,
                    color=(255, 255, 255),
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )

    def _draw_polygon(self, img):
        """
            Draw polygon of surface projection matrix area in the source image
            to calibrate extrinsic
            Args:
                img: 'cv2.math' image to perform operations and draw elements
            returns:
        """

        # Draw a line if only two points
        if len(self.pts) == 2:
            cv2.line(
                img=img,
                pt1=self.pts[0],
                pt2=self.pts[1],
                color=(255, 0, 0),
                thickness=2,
            )

        # Draw a polygon if more than 2 points
        elif len(self.pts) >= 3:
            pts = np.array(self.pts, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(
                img=img, pts=[pts], isClosed=True, color=(255, 0, 0), thickness=2
            )

    def _draw_rulers(self, img, pose):
        """
            Draw rulers for mouse position on extrinsic calibrator window
            Args:
                img: 'cv2.math' image to perform operations and draw elements
                pose: 'tuple' current XY mouse position over the image
            returns:
        """

        # Draw horizontal line
        dotline(
            src=img,
            p1=(0, pose[1]),
            p2=(img.shape[1], pose[1]),
            color=(0, 0, 255),
            thickness=1,
            Dl=5,
        )
        # Draw vertical line
        dotline(
            src=img,
            p1=(pose[0], 0),
            p2=(pose[0], img.shape[0]),
            color=(0, 0, 255),
            thickness=1,
            Dl=5,
        )

    def _draw_skyview(self):
        """
            Draw bird/sky view from projection matrix
            Args:
            returns:
        """

        # Proceed only if there's projection matrix
        if self._M is not None:

            # Warp perspective of source/original image to get bird view
            self.img_sky = cv2.warpPerspective(
                dsize=(self.warp_width, self.warp_height), src=self.img_src, M=self._M
            )

            # -----------------------------------------------------------------
            # GUI and visuals

            # Draw linners
            for i in range(0, self.warp_width, self._WIN_WARP_VER_DIV):
                cv2.line(
                    img=self.img_sky,
                    pt1=(i, 0),
                    pt2=(i, self.warp_height),
                    color=(0, 255, 0),
                    thickness=1,
                )
            for i in range(0, self.warp_height, self._WIN_WARP_HOZ_DIV):
                cv2.line(
                    img=self.img_sky,
                    pt1=(0, i),
                    pt2=(self.warp_width, i),
                    color=(0, 255, 0),
                    thickness=1,
                )

            pts = np.array(
                (
                    self._Mdst_pts["p1"],
                    self._Mdst_pts["p2"],
                    self._Mdst_pts["p3"],
                    self._Mdst_pts["p4"],
                ),
                np.int32,
            )
            cv2.polylines(
                img=self.img_sky,
                pts=[pts],
                isClosed=True,
                color=(255, 0, 0),
                thickness=2,
            )
            cv2.polylines(
                img=self.img_sky,
                pts=[np.array(self._MOriginpts)],
                isClosed=True,
                color=(255, 255, 0),
                thickness=2,
            )

            # Draw surface lines
            dotline(
                src=self.img_sky,
                p1=self._Mdst_pts["p1"],
                p2=self._Mdst_pts["p3"],
                color=(255, 255, 255),
                thickness=1,
                Dl=5,
            )
            dotline(
                src=self.img_sky,
                p1=self._Mdst_pts["p2"],
                p2=self._Mdst_pts["p4"],
                color=(255, 255, 255),
                thickness=1,
                Dl=5,
            )

            # Draw matrix surfarce projection points
            for key, pt in self._Mdst_pts.items():
                if pt is not None:
                    cv2.circle(
                        img=self.img_sky,
                        center=pt,
                        radius=3,
                        color=(0, 0, 255),
                        thickness=-1,
                    )
                    cv2.putText(
                        img=self.img_sky,
                        text="{}".format(key),
                        org=(pt[0] - 8, pt[1] - 8),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.7,
                        color=(0, 0, 0),
                        thickness=3,
                        lineType=cv2.LINE_AA,
                    )
                    cv2.putText(
                        img=self.img_sky,
                        text="{}".format(key),
                        org=(pt[0] - 8, pt[1] - 8),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.7,
                        color=(255, 255, 255),
                        thickness=1,
                        lineType=cv2.LINE_AA,
                    )

            self._draw_rulers(img=self.img_sky, pose=self.bird_mouse_current_pos)

            self._draw_measure(img=self.img_sky)

            # -----------------------------------------------------------------

        self._win_bird_width = self.img_sky.shape[1]
        self._win_bird_height = self.img_sky.shape[0]
        hfscl = self._win_bird_width / self._win_bird_max_width
        vfscl = self._win_bird_height / self._win_bird_max_height
        if vfscl > hfscl:
            self.img_sky = image_resize(
                image=self.img_sky,
                width=None,
                height=int(self._win_bird_max_height),
                inter=cv2.INTER_AREA,
            )
        else:
            self.img_sky = image_resize(
                image=self.img_sky,
                width=int(self._win_bird_max_width),
                height=None,
                inter=cv2.INTER_AREA,
            )

        # Plot and show window
        cv2.imshow(self._win_bird_name, self.img_sky)
        cv2.waitKey(self._win_bird_time)

    def _update_pix_relation(self, dm, dp, relation):
        """
            updates pixels per meter relation
            Args:
                dm: 'float' measure in meters
                dp: 'int' measure in pixels
                relation: 'string' relation
                    dv: for vertical measurement
                    dh: for horizontal measurement
            returns:
        """

        if relation == "dv":
            self.ppmy = float(dm / dp)
            printlog(
                msg="Vertical relation updated, new values are "
                "ppmx={:.2f}[pix/m], ppmy={:.2f}[pix/m]".format(self.ppmx, self.ppmy),
                msg_type="USER",
            )
        elif relation == "dh":
            self.ppmx = float(dm / dp)
            printlog(
                msg="Horizontal relation updated, new values are "
                "ppmx={:.2f}[pix/m], ppmy={:.2f}[pix/m]".format(
                    self.ppmx if self.ppmx is not None else 0.0,
                    self.ppmy if self.ppmy is not None else 0.0,
                ),
                msg_type="USER",
            )
        else:
            printlog(msg="No valid relation to update", msg_type="ERROR")

    def _load_extrinsic(self):
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
                self._Mpts = {
                    "p1": tuple(data["Mpt_src"]["p1"]),
                    "p2": tuple(data["Mpt_src"]["p2"]),
                    "p3": tuple(data["Mpt_src"]["p3"]),
                    "p4": tuple(data["Mpt_src"]["p4"]),
                }
                self.pts = [tuple(pt) for pt in self._Mpts.values()]
                self._VISION_CAL_WARPED_SIZE = tuple(data["M_warped_size"])
                self.img_scl = data["img_scl"]
                # self.img_width = data["src_img_size"][0]
                # self.img_height = data["src_img_size"][1]
                self.ppmx = data["ppmx"]
                self.ppmy = data["ppmy"]
                self.warp_width = data["dst_warp_size"][0]
                self.warp_height = data["dst_warp_size"][1]
                self._Mdst_pts = {
                    "p1": tuple(data["Mdst_pts"]["p1"]),
                    "p2": tuple(data["Mdst_pts"]["p2"]),
                    "p3": tuple(data["Mdst_pts"]["p3"]),
                    "p4": tuple(data["Mdst_pts"]["p4"]),
                }
                self._in_gerion = data["in_region"]

                self._calibrate_extrinsic()

                # Update window elements
                self._win_bird_event = True
                self._win_event = True

                printlog(
                    msg="Extrinsic file {} loaded".format(self._ext_file),
                    msg_type="INFO",
                )

            except IOError as e:  # Report any error saving de data
                printlog(msg="Error loading extrinsic, {}".format(e), msg_type="ERROR")
        else:
            printlog(msg="No extrinsic file {}".format(self._ext_file), msg_type="WARN")

    def _save_extrinsic(self):
        """
            Save extrinsic calibration parameters in a yaml file in extrinsic 
            calibration folder and path defined in constructor
            Args:
            returns:
        """

        # Don't proceed if there's no calibration matrix
        if self._M is None:
            printlog(msg="No transformation matrix yet to save", msg_type="WARN")
            return

        # Get dictionary of extrinsic parameters to save in yaml file
        ext_params = {
            "M": self._M.tolist(),
            "Mpt_src": {key: list(pt) for key, pt in self._Mpts.items()},
            "Mdst_pts": {key: list(pt) for key, pt in self._Mdst_pts.items()},
            "ppmx": self.ppmx,
            "ppmy": self.ppmy,
            "M_warped_size": list(self._VISION_CAL_WARPED_SIZE),
            "src_img_size": [self.img_width, self.img_height],
            "dst_warp_size": [self.warp_width, self.warp_height],
            "img_scl": self.img_scl,
            "intrinsic": False if self._intrinsic is None else True,
            "in_region": self._in_gerion,
        }

        # Absolute path to save file
        file_path = os.path.join(self._ext_path, self._ext_file)
        try:  # Save the calibration data in file
            with open(file_path, "w") as outfile:
                yaml.dump(ext_params, outfile, default_flow_style=False)
            printlog(
                msg="Extrinsic parameters saved {}".format(file_path), msg_type="INFO"
            )
        except IOError as e:  # Report any error saving de data
            printlog(msg="Error saving extrinsic, {}".format(e), msg_type="ERROR")

    def _calibrate_extrinsic(self):
        """
            Description
            Args:
                variable: 'type' description
            returns:
                variable: 'type' description
        """

        # Check if there're all user's points assigned to perform extrinsic
        # calibration
        for pt in self._Mpts.values():
            if pt is None:
                printlog(
                    msg="No transformation matrix yet", msg_type="WARN"
                )
                return

        # Get space transformation matrix
        M = get_rot_matrix(
            self._Mpts["p1"],
            self._Mpts["p2"],
            self._Mpts["p3"],
            self._Mpts["p4"],
            self._VISION_CAL_WARPED_SIZE,
        )

        # Get image size coordinates in warped space
        dst_pt_ltop = get_projection_point_dst(coords_src=(0, 0), M=M)
        dst_pt_rtop = get_projection_point_dst(coords_src=(self.img_width, 0), M=M)
        dst_pt_lbottom = get_projection_point_dst(coords_src=(0, self.img_height), M=M)
        dst_pt_rbottom = get_projection_point_dst(
            coords_src=(self.img_width, self.img_height), M=M
        )
        self._MOriginpts = [dst_pt_ltop, dst_pt_rtop, dst_pt_rbottom, dst_pt_lbottom]

        if not self._in_gerion:
            # Get bounding rect with the new geometry
            x, y, self.warp_width, self.warp_height = cv2.boundingRect(
                np.array((dst_pt_ltop, dst_pt_rtop, dst_pt_lbottom, dst_pt_rbottom))
            )

            # Apply translation component to rotation matrix
            Mt = np.float32([[1.0, 0, -float(x)], [0, 1.0, -float(y)], [0, 0, 1.0]])
            self._M = np.matmul(Mt, M)

            # Get image size coordinates in warped space
            dst_pt_ltop = get_projection_point_dst(coords_src=(0, 0), M=self._M)
            dst_pt_rtop = get_projection_point_dst(
                coords_src=(self.img_width, 0), M=self._M
            )
            dst_pt_lbottom = get_projection_point_dst(
                coords_src=(0, self.img_height), M=self._M
            )
            dst_pt_rbottom = get_projection_point_dst(
                coords_src=(self.img_width, self.img_height), M=self._M
            )
            self._MOriginpts = [
                dst_pt_ltop,
                dst_pt_rtop,
                dst_pt_rbottom,
                dst_pt_lbottom,
            ]

        else:
            self.warp_width = self._VISION_CAL_WARPED_SIZE[0]
            self.warp_height = self._VISION_CAL_WARPED_SIZE[1]
            self._M = M

        # Get image size coordinates in warped space
        dst_p1 = get_projection_point_dst(coords_src=self._Mpts["p1"], M=self._M)
        dst_p2 = get_projection_point_dst(coords_src=self._Mpts["p2"], M=self._M)
        dst_p3 = get_projection_point_dst(coords_src=self._Mpts["p3"], M=self._M)
        dst_p4 = get_projection_point_dst(coords_src=self._Mpts["p4"], M=self._M)
        self._Mdst_pts = {"p1": dst_p1, "p2": dst_p2, "p3": dst_p3, "p4": dst_p4}

        # print(self.img_src.shape, self.img_width, self.img_height)
        self._win_bird_event = True

    def _rotate_space(self):
        """
            Description: Rotate transformation space
            Args:
            returns:
        """

        # Right points rotation
        self._Mpts["p1"], self._Mpts["p2"], self._Mpts["p3"], self._Mpts["p4"] = (
            self._Mpts["p4"],
            self._Mpts["p1"],
            self._Mpts["p2"],
            self._Mpts["p3"],
        )

        # Update windows
        self._calibrate_extrinsic()
        self._win_event = True
        self._win_bird_event = True

    def _assing_mpt(self, pt_idx):
        """
            Assing surface projection point to current user's point 
            Args:
                pt_idx: 'int' user's point index 
            returns:
        """

        pt = self.pts[self.pts_idx]

        for key, Mpt in self._Mpts.items():
            if Mpt is not None:
                if pt[0] == Mpt[0] and pt[1] == Mpt[1]:
                    self._Mpts[key] = None

        self._Mpts["p{}".format(pt_idx)] = self.pts[self.pts_idx]

    def _move_pt(self, direction="up"):
        """
            Move current user point depending on the option 
                up, down, left, right
            Args:
            returns:
        """

        if len(self.pts):
            if direction == "up":
                self.pts[self.pts_idx] = (
                    self.pts[self.pts_idx][0],
                    self.pts[self.pts_idx][1] - 1,
                )
            elif direction == "down":
                self.pts[self.pts_idx] = (
                    self.pts[self.pts_idx][0],
                    self.pts[self.pts_idx][1] + 1,
                )
            elif direction == "left":
                self.pts[self.pts_idx] = (
                    self.pts[self.pts_idx][0] - 1,
                    self.pts[self.pts_idx][1],
                )
            elif direction == "right":
                self.pts[self.pts_idx] = (
                    self.pts[self.pts_idx][0] + 1,
                    self.pts[self.pts_idx][1],
                )
            else:
                return

            # Find the closet neighbor to re-assing surface projection point
            # if there's one
            key, _ = closest_neighbor(
                self._Mpts, x=self.pts[self.pts_idx][0], y=self.pts[self.pts_idx][1]
            )
            if key is not None:
                self._Mpts[key] = (self.pts[self.pts_idx][0], self.pts[self.pts_idx][1])

            # Update window components
            self._calibrate_extrinsic()
            self._win_event = True
            self._win_bird_event = True

    def draw_gui(self):
        """
            Draw extrinsic calibration window elements
            Args:
            returns:
        """

        if self._show_menu:
            cv2.imshow(self._win_name, self._img_menu)
        else:
            if self._win_event:

                self.img_gui = self.img_src.copy()

                # Draw gui window elements
                self._draw_polygon(img=self.img_gui)
                self._draw_points(img=self.img_gui)
                self._draw_rulers(img=self.img_gui, pose=self.mouse_current_pos)

                self._win_event = False

            if self._win_bird_event:
                if self._M is not None:
                    self._draw_skyview()
                    self._win_bird_event = False

            cv2.imshow(self._win_name, self.img_gui)
        key = cv2.waitKey(self._win_time)

        # If no key pressed do nothing
        if key == -1:
            pass
        # If Q key pressed then save extrinsic
        elif key == 113 or key == 81:
            self._save_extrinsic()
            exit()
        # If S key pressed then save extrinsic
        elif key == 115:
            self._save_extrinsic()
        # If L key pressed then save extrinsic
        elif key == 108:
            self._load_extrinsic()
        # If A key pressed then go previous point
        elif key == 97:
            self._win_event = True
            if self.pts_idx + 1 <= len(self.pts) - 1:
                self.pts_idx += 1
            elif self.pts_idx == len(self.pts) - 1:
                self.pts_idx = 0
        # If D key pressed then next previous point
        elif key == 100:
            self._win_event = True
            if self.pts_idx > 0:
                self.pts_idx -= 1
            elif self.pts_idx == 0:
                self.pts_idx = len(self.pts) - 1
        #  If R key pressed then rotate space
        elif key == 114:
            self._rotate_space()
        # If U key pressed then move current point idx up
        elif key == 117:
            self._move_pt(direction="up")
        # If J key pressed then move current point idx down
        elif key == 106:
            self._move_pt(direction="down")
        # If K key pressed then move current point idx to the right
        elif key == 107:
            self._move_pt(direction="right")
        # If TAB key pressed then show menu/help
        elif key == 9:
            self._show_menu = not self._show_menu
        # If H key pressed then move current point idx to the left
        elif key == 104:
            self._move_pt(direction="left")
        # If 1, 2, 3, or 4 key pressed then assing point to pt
        elif key == 49 or key == 50 or key == 51 or key == 52:
            self._assing_mpt(pt_idx=chr(key))
            self._win_event = True
        # If E key pressed then performe extrinsic calibration
        elif key == 101 or self._win_bird_event:
            self._calibrate_extrinsic()
        # If I key pressed then analyse in region
        elif key == 105:
            self._in_gerion = not self._in_gerion
            self._calibrate_extrinsic()
        # If no key action found
        else:
            printlog(msg="key {} command no found".format(key), msg_type="WARN")


# =============================================================================
def main(argv):

    # -------------------------------------------------------------------------
    extrinsic_file = None
    intrinsic_file = None
    source_file = None
    intrinsic = None

    # -------------------------------------------------------------------------
    # Read user input
    try:
        opts, _ = getopt.getopt(argv, "hv:e:i:", ["video=", "extfile=", "intfile="])
    except getopt.GetoptError:
        printlog(
            "calibrate_extrinsic.py -v <video_source_file> -e "
            "<extrinsic_file> -i <intrinsic_file>"
        )
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            printlog(
                "calibrate_extrinsic.py -v <video_source_file> -e "
                "<extrinsic_file> -i <intrinsic_file>"
            )
            sys.exit()
        elif opt in ("-e", "--extrinsic"):
            extrinsic_file = arg
        elif opt in ("-i", "--intrinsic"):
            intrinsic_file = arg
        elif opt in ("-v", "--video"):
            source_file = arg

    # -------------------------------------------------------------------------
    # Get source to calibrate extrinsic
    fwd = os.path.dirname(os.path.abspath(__file__))
    fwd = os.path.abspath(os.path.join(fwd, os.pardir))

    # Take first file in media folder if and input was not defined
    if source_file is None:
        files = glob.glob("./media/*.mp4")
        if not len(files):
            printlog(msg="File {} no found", msg_type="ERROR")
            return 2
        file_default = os.path.basename(files[0])

    file_name = file_default if source_file is None else source_file
    file_ext = file_name.split(".")[-1]
    file_path = os.path.join(fwd, "media", file_name)

    if not os.path.isfile(file_path):
        printlog(msg="File {} no found".format(file_name), msg_type="ERROR")
        return 2

    # If source file is a video
    if file_ext == "mp4":

        # Create a VideoCapture object and read from input file
        # If the input is the camera, pass 0 instead of the video file name
        cap = cv2.VideoCapture(file_path)

        # Check if camera opened successfully
        if not cap.isOpened():
            printlog(msg="Error opening video stream or file", msg_type="ERROR")

        # Get video frame
        ret, frame = cap.read()

        # Check if camera opened successfully
        if not ret:
            printlog(msg="Error getting video frame", msg_type="ERROR")

    # If source file is an image
    elif file_ext == "jpg" or file_ext == "png":
        frame = cv2.imread(file_path)

    # If file has a invalid format
    else:
        printlog(msg="File {} with invalid".format(file_name), msg_type="ERROR")
        return 2

    # -------------------------------------------------------------------------
    # Load intrinsic parameters
    if intrinsic_file is not None:
        intrinsic_path = os.path.join(fwd, "configs", intrinsic_file)
        intrinsic = read_intrinsic_params(
            CONF_PATH=os.path.join(fwd, "configs"), FILE_NAME=intrinsic_file
        )
    else:
        printlog(msg="No using intrinsic file", msg_type="WARN")

    # -------------------------------------------------------------------------
    # Create and run extrinsic calibrator object
    extrinsic = calibrator(extrinsic=extrinsic_file, intrinsic=intrinsic, img=frame)
    while True:
        extrinsic.draw_gui()


# =============================================================================
if __name__ == "__main__":
    main(sys.argv[1:])

# =============================================================================
