#!/usr/bin/env python3
# =============================================================================
import sys

# For user with ros1 installed in their system
cv2ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if cv2ros_path in sys.path:
    sys.path.remove(cv2ros_path)

import numpy as np
import getopt
import glob
import yaml
import cv2
import os

sys.path.append(".") 
from python_utils.vision_utils import closest_neighbor
from python_utils.vision_utils import get_rot_matrix
from python_utils.vision_utils import printlog
from python_utils.vision_utils import dotline

# =============================================================================
class calibrator():

    def __init__(self):
        """
            Intrinsic calibrator constructor
            Args: NaN
            Returns: Nan
        """

        # Interface
        self._win_name = "intrinsic_calibrator"

        # Set Path to save intrinsic file
        self._int_file = 'intrinsic.yaml'
        fwd = os.path.dirname(os.path.abspath(__file__))
        fwd = os.path.abspath(os.path.join(fwd, os.pardir))
        self._int_path = os.path.join(fwd, "configs")
        self._imgs_path = os.path.join(fwd, "calibration/chess_board")

        # Images
        self._img_src = None
        self._win_time = 50
        self._frame = 0
        
        # Camera calibration values
        self._nCornX = int(os.getenv(key="MONO_PATTERN_HOZ", default=6))
        self._nCornY = int(os.getenv(key="MONO_PATTERN_VER", default=4))
        self._curr_frame = 0
        self._img_size = None
        self._obj_pts = None
        self._cal_imgs = []
        
        # Camera calibration matrices
        self._rep_error = 0.0
        self._camMat = None
        self._distCoef = None
        self._rotVect = None
        self._transVect = None   

        # Intrinsic calibration data
        self._cal_data = None
    
    def _write_text(self, image, text, coords):
        """
        Function to write on the top of the images
        Args: Image, Text [String], Coords [X, Y] 
        Returns: Image (Containing the given text)
        """

        image = cv2.putText(img=image, 
            text=text, 
            org=coords, 
            fontFace=cv2.FONT_HERSHEY_DUPLEX, 
            fontScale=0.7, 
            color=(0, 255, 0), 
            thickness=2)
        
        return image
        
    def _video_capture(self, img):
        """
            Desc
            Args:
            returns:
        """
        
        self._img_src = self._write_text(
            image=img,
            text="Press 'S' to save the current frame.", 
            coords=(25, 45))
        
        self._img_src = self._write_text(
            image=self._img_src,
            text="Press 'Q' to exit.", 
            coords=(25, 80))
        
        cv2.imshow(self._win_name, self._img_src)    
        key = cv2.waitKey(self._win_time)

        if key == -1:
            pass
        # If Q key pressed then close the window
        elif key == 113 or key == 81: 
            cv2.destroyAllWindows()
            exit()
        # If S key pressed then save current frame
        elif key == 115:
            self._save_frame()
        # Invalid key
        else:
            printlog(msg="key {} command no found".format(key), 
                msg_type="WARN")
    
    def calibrate(self, imgs_path):
        """
            Function to perform the intrinsic camera calibration
            Args: NaN
            Returns: NaN
        """
        
        # Loading images for calibration
        self._load_frames(imgs_path) 
        
        # Object points for chessboard pattern
        self._create_objPts()
        
        object_pts = []
        image_pts = []
        n_frame = 0
        wrong_cal = 0
        
        printlog(msg="Starting calibration.",
                msg_type="INFO")
        
        for frame in self._cal_imgs:
            
            # Frame reading
            src_img = cv2.imread(filename=frame)
            
            # Color transformation
            gray_img = cv2.cvtColor(src=src_img, code=cv2.COLOR_RGB2GRAY)
            
            # Finding chessboard inner corners
            ret, corners = cv2.findChessboardCorners(image=gray_img, 
                patternSize=(self._nCornX, self._nCornY))
                        
            if ret == True:
                self._curr_frame += 1
                
                # Points appending for calibration
                object_pts.append(self._obj_pts)    
                image_pts.append(corners)
                
                # Drawing the chessboard corners in the current frame
                img_chess = cv2.drawChessboardCorners(image=src_img,
                    patternSize=(self._nCornX, self._nCornY), 
                    corners=corners, 
                    patternWasFound=ret)
                
                img_chess = self._write_text(
                    image=img_chess, 
                    text="Frame {}".format(self._curr_frame), 
                    coords=(25, 45))
                
                cv2.imshow("Chessboard detections", img_chess)
                key = cv2.waitKey(100)
                
                if key == 113 or key == 81: 
                    cv2.destroyAllWindows()
                    exit()
            
            else: 
                wrong_cal += 1
                
            n_frame += 1
            
        self._img_size = (src_img.shape[1], src_img.shape[0])
        
        # Performing calibration
        self._rep_error, self._camMat, self._distCoef, self._rotVect, self._transVect = \
            cv2.calibrateCamera(object_pts, image_pts, self._img_size, None, None)   
            
        printlog(msg="Resulting reprojection error: %.5f" % self._rep_error, 
                msg_type="INFO")
        
        if (self._rep_error >= 0.5):
            printlog(msg="Consider adding more images of the pattern, to reduce the Reprojection Error", 
                msg_type="WARN")
        
        if (wrong_cal != 0):
            printlog(msg="Calibration not found in {} images.". format(wrong_cal), 
                msg_type="WARN")
        
        printlog(msg="Calibration completed.", msg_type="OKGREEN")
        
        self._save_calibration()
        
    def _create_objPts(self):
        """
            Function to create the chessboard matrix points for calibration
            Args: NaN
            Returns: NaN
        """
        printlog(msg="Creating object points", msg_type="INFO")

        self._obj_pts = np.zeros((self._nCornX * self._nCornY, 3), np.float32)
        self._obj_pts[:, :2] = np.mgrid[0:self._nCornX, 0:self._nCornY].T.reshape(-1, 2)
    
    def _load_frames(self, imgs_path):
        """
            Function to load the chessboard images for calibration 
            Args: NaN
            Returns: NaN
        """
        
        # Defining images path files
        self._imgs_path = imgs_path
        
        printlog(msg="Loading calibration images ...", 
                msg_type="INFO")
        
        self._cal_imgs = glob.glob(self._imgs_path + "/*.jpg")
        
        if len(self._cal_imgs) == 0:
            printlog(msg="{} folder is empty,"
                " please allocate the calibration images there.".format(self._imgs_path),
                msg_type="ERROR")
            exit()
        else:
            printlog(msg="{} Images were loaded.".format(len(self._cal_imgs)), 
            msg_type="OKGREEN")
            
    def _save_calibration(self):
        """
            Function to save the calibration file in .yaml format
            Args: NaN
            Returns: NaN
        """
        # Dictionary containing the data
        data = {
            "image_height": self._img_size[0],
            "image_width": self._img_size[1],
            "camera_matrix":{
                "rows": 3,
                "cols": 3,
                "data": list(self._camMat.tolist())
                },
            "distortion_model": "rational_polynomial",
            "distortion_coefficients":{
                "rows": 1,
                "cols": 5,
                "data": list(self._distCoef.tolist())
                }
            }

        file_path = os.path.join(self._int_path, self._int_file)
        
        with open(file_path, "w") as output_file: 
            yaml.dump(data, output_file, default_flow_style=False, sort_keys=False)
            
        printlog(msg="Intrinsic calibration saved successfuly at {}". format(file_path), 
                msg_type="OKGREEN")
        
    def _save_frame(self):
        """
            Function to save the current frame of the video
            Args: NaN
            Returns: NaN
        """
        
        cv2.imwrite(os.path.join(self._imgs_path, "chessboard_{}.jpg".format(self._frame)), \
            self._img_src)
        self._frame += 1
        
        printlog(msg="Frame saved", msg_type="INFO")

# =============================================================================
def main(argv):
    # Reading input arguments
    try:
        opts, args = getopt.getopt(argv,"hi:", 
            ["path="])
    except getopt.GetoptError:
        printlog('calibrate_extrinsic.py -i <intrinsic_action>')
        sys.exit(2)
        
    for opt, arg in opts:
        if opt == '-h':
            printlog('calibrate_extrinsic.py -i ' 
                '<video name, imgs path or video device>')
            sys.exit()
        elif opt in ("-i", "--intrinsic"):
            intrinsic_file = arg
    
    # -------------------------------------------------------------------------
    # Calibrator instantiation
    intrinsic = calibrator()
    
    # Get source to calibrate extrinsic
    fwd = os.path.dirname(os.path.abspath(__file__))
    fwd = os.path.abspath(os.path.join(fwd, os.pardir))
    
    if (len(intrinsic_file) < 4):
        videoDevice = int(intrinsic_file)
        cap = cv2.VideoCapture(videoDevice)
        
        if not cap.isOpened():
            printlog(msg="Error opening video device, "
                "please try a different one",
                msg_type="ERROR")
            sys.exit()

        while True:
            ret, frame = cap.read()
            
            if not ret: 
                printlog(msg="Error getting video frame",
                    msg_type="ERROR")
                sys.exit()
            
            intrinsic._video_capture(frame)
    
    elif (len(intrinsic_file) >= 4):
        file_ext = str(intrinsic_file).split(".")
        file_name = str(intrinsic_file)
        
        if (len(file_ext) == 1):
            imgs_path = os.path.join(fwd, "calibration" , file_name)
            intrinsic.calibrate(imgs_path)
            
        elif (len(file_ext) == 2):
            if ((file_ext[1] == "mp4") or (file_ext[1] == "avi")):
                file_path = os.path.join(fwd, "media", file_name)
                
                # Openning video file
                cap = cv2.VideoCapture(file_path)
                
                # Check if camera opened successfully
                if not cap.isOpened():
                    printlog(msg="Error opening video stream or file",
                        msg_type="ERROR")
                    sys.exit()

                while True:
                    ret, frame = cap.read()
                    
                    if not ret: 
                        printlog(msg="Error getting video frame",
                            msg_type="ERROR")
                        sys.exit()
                        
                    intrinsic._video_capture(frame)
                
            else:
                printlog(msg="Invalid format. A .mp4 or .avi file is required",
                    msg_type="ERROR")
                sys.exit()
            
        else: 
            printlog(msg="Not valid argument",
                msg_type="ERROR")
            sys.exit()
        
if __name__ == "__main__":
    main(sys.argv[1:])
