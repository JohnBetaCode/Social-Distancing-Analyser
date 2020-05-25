#!/usr/bin/env python3
# =============================================================================
import numpy as np
import time
import yaml
import sys
import cv2
import os

sys.path.append("./python_utils")
from vision_utils import printlog
from vision_utils import print_text_list

# =============================================================================
class Player:
    def __init__(self, win_name, video_src_file, media_loop=True):
        """
            Class constructor initialization
            Args:
                win_name: 'string' name of the player window
                video_src: 'string' description file name with media sources
                media_loop: 'boolean' Enable/Disable media looping
            returns:
        """

        self._WIN_MAX_WIDTH = int(os.getenv(key="WIN_WIDTH", default=640))
        self._WIN_MAX_HEIGHT = int(os.getenv(key="WIN_HEIGHT", default=480))
        self._RESIZE_FLAG = cv2.INTER_NEAREST

        # player propierties
        self._player_media_loop = media_loop

        # media player attributes
        self._win_name = win_name
        # Load menu/help/options image
        self._win_menu = cv2.imread(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "figures", "player_menu.jpg"
            )
        )
        self._win_show_menu = True
        self._win_pause = True
        self._win_img = np.zeros(
            (self._WIN_MAX_HEIGHT, self._WIN_MAX_WIDTH, 3), np.uint8
        )
        cv2.namedWindow(self._win_name)
        self.tick = time.time()

        # Data source videos attributes
        self._src_path = os.path.split(video_src_file)[0]
        self._src_list = []
        self._read_media_source(media_list=video_src_file)

        # Capture attributes
        self.cap_img = np.zeros(
            (self._WIN_MAX_HEIGHT, self._WIN_MAX_WIDTH, 3), np.uint8
        )
        self._cap = None  # Video capture cv2 object
        self._cap_ret = None  # Video retention
        self._cap_idx = 0  # Current video capture index
        self._cap_frame_msec = 0  # Current video frame in mil seconds
        self._cap_frame = 0  # Current video frame number
        self._cap_frame_avi_ratio = 0  # Frame/Video ratio
        self._cap_frame_width = 0  # Frame/Video width
        self._cap_frame_height = 0  # Frame/video height
        self._cap_rate = 1.0  # Frame rate
        self._cap_fourcc = None  # Video codec
        self._cap_frame_count = 0  # Number of frames in video

        # Current file parameters from yaml conf file
        self.file_name = None
        self.file_intrinsic = None
        self.file_extrinsic = None

        # Start player
        self._start_capture()

    def _read_media_source(self, media_list):
        """
            read video media sources specified in file media_list located in
            root folder /media
            Args:
                media_list: 'string' media sources file name
            returns:
        """

        # Get file absolute path to media sourcesW
        fwd = os.path.dirname(os.path.abspath(media_list))
        file_path = os.path.join(fwd, os.path.basename(media_list))

        try:
            # Check file existence
            if os.path.isfile(file_path):
                with open(file_path, "r") as stream:
                    self._src_list = yaml.safe_load(stream)
                if not len(self._src_list):
                    printlog(
                        msg="No elements in {}".format(media_list), msg_type="ERROR"
                    )
                    exit()
            else:
                printlog(msg="file no found", msg_type="ERROR")
                exit()
        # Report any error saving de data
        except IOError as e:
            printlog(msg="Error loading data list, {}".format(e), msg_type="ERROR")

        printlog(msg="Data list loaded from {}".format(media_list), msg_type="OKGREEN")

        # Print video sources in list to be reproduced
        for data_src in self._src_list:
            print(
                "\tfile:{}    \tExtrinsic:{}  \tIntrinsics".format(
                    data_src["file_name"], data_src["extrinsic"], data_src["intrinsic"]
                )
            )

    def _start_capture(self, cap_idx=0):
        """
            start video capture in index "cap_idx"
            Args:
                cap_idx: 'int' video capture in media sources list to start
            returns:
        """

        self.file_name = file_name = self._src_list[cap_idx]["file_name"]

        # Check video file existence
        if not os.path.isfile(os.path.join(self._src_path, file_name)):
            printlog(msg="file video {} no found".format(file_name), msg_type="ERROR")
            exit()
            return

        # Start video capture
        self._cap = cv2.VideoCapture(os.path.join(self._src_path, file_name))

        # Check if video was opened successfully
        if not self._cap.isOpened():
            printlog(
                msg="Error opening video stream or file {}".format(
                    self._src_list[cap_idx]["file_name"]
                ),
                msg_type="ERROR",
            )
            exit()

        self.file_intrinsic = self._src_list[cap_idx]["intrinsic"]
        self.file_extrinsic = self._src_list[cap_idx]["extrinsic"]

        # Get video properties
        self._cap_frame_avi_ratio = self._cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self._cap_frame_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._cap_frame_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._cap_rate = self._cap.get(cv2.CAP_PROP_FPS)
        self._cap_fourcc = self._cap.get(cv2.CAP_PROP_FOURCC)
        self._cap_frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # print video properties
        # printlog(msg="FILE: {} - SIZE:{}X{} - FPS:{} - FRAMES:{}"
        #     " - DURATION:{} [sec]".format(
        #         file_name,
        #         self._cap_frame_width,
        #         self._cap_frame_height,
        #         round(self._cap_rate, 2),
        #         self._cap_frame_count,
        #         round(self._cap_frame_count/self._cap_rate), 2),
        #         msg_type="USER")

    def _get_frame(self):
        """
            get video frame in current video capture
            Args:
            returns:
        """

        # If video already finished
        if self._cap_frame == self._cap_frame_count - 1:
            if self._player_media_loop:
                if len(self._src_list) == 1:
                    self._cap_idx = 0
                elif self._cap_idx < len(self._src_list) - 1:
                    self._cap_idx += 1
                else:
                    self._cap_idx = 0
                self._start_capture(cap_idx=self._cap_idx)

            else:
                return
        if self._cap is None:
            self._player_media_loop = False
            return

        # Get video frame
        self._cap_ret, self.cap_img = self._cap.read()

        # Get video properties
        self._cap_frame = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))
        self._cap_frame_msec = self._cap.get(cv2.CAP_PROP_POS_MSEC)

        # Check if camera opened successfully
        if not self._cap_ret:
            printlog(
                msg="Error getting video frame in {}".format(
                    self._src_list[self._cap_idx]["file_name"]
                ),
                msg_type="ERROR",
            )
            exit()

    def _draw_gui(self, img):
        """
            Draw video capture information to show in player window image
            Args:
                img: 'cv2.math' image to draw components
            returns:
        """

        # Info to show in player window image
        text_list = [
            "({}/{}) {}".format(
                self._cap_idx + 1,
                len(self._src_list),
                self._src_list[self._cap_idx]["file_name"],
            ),
            "{:.2f}% - {}/{}".format(
                self._cap_frame / self._cap_frame_count * 100,
                self._cap_frame_count,
                self._cap_frame,
            ),
            "size: {}x{}".format(self._cap_frame_width, self._cap_frame_height),
            "fps: {}".format(round(self._cap_rate, 2)),
        ]

        print_text_list(
            img=img,
            tex_list=text_list,
            color=(255, 255, 255),
            orig=(10, 25),
            fontScale=0.4,
        )

    def _key_action(self, key):
        """
            execute user input action by pressed key
            Args:
                key: 'int' cv2 key input key to execute action
            returns:
        """

        # If no key pressed do nothing
        if key == -1:
            pass
        # If Q key pressed then Quit program
        elif key == 113:
            exit()
        # If L key pressed then Start video again
        elif key == 108:
            self._start_capture(cap_idx=self._cap_idx)
        # If R key pressed record video
        elif key == 114:
            pass
        # If Space key pressed then pause/reproduce video
        elif key == 32:
            self._win_pause = not self._win_pause
        # If N key pressed then go to the next video
        elif key == 110:
            if len(self._src_list) == 1:
                self._cap_idx = 0
            elif self._cap_idx < len(self._src_list) - 1:
                self._cap_idx += 1
            else:
                self._cap_idx = 0
            self._start_capture(cap_idx=self._cap_idx)
        # If B key pressed then go to previous video
        elif key == 98:
            if len(self._src_list) == 1:
                self._cap_idx = 0
            elif self._cap_idx > 0:
                self._cap_idx -= 1
            elif self._cap_idx == 0:
                self._cap_idx = len(self._src_list) - 1
            self._start_capture(cap_idx=self._cap_idx)
        # If C key pressed then take snapshot
        elif key == 99:
            self.snapshot()
        # If Tab key pressed then take snapshot
        elif key == 9:
            self._win_show_menu = not self._win_show_menu
            self._win_pause = self._win_show_menu

        # If no key action found
        else:
            printlog(msg="key {} command no found".format(key), msg_type="WARN")

    def snapshot(self):
        """
            take and save a snapshot of the current video frame in the video
            capture media player
            Args:
            returns:
        """

        # Create captures folder in ./media if doesn't exit one
        capture_path = os.path.join(self._src_path, "captures")
        if not os.path.exists(capture_path):
            os.mkdir(capture_path)

        # Get capture image file name
        capture_name = "{}_cap{}.jpg".format(
            os.path.splitext(self._src_list[self._cap_idx]["file_name"])[0],
            self._cap_frame,
        )

        # Write image in ./media/captures folder
        cv2.imwrite(os.path.join(capture_path, capture_name), self.cap_img)
        printlog(
            msg="capture saved as {} in capture folder".format(capture_name),
            msg_type="INFO",
        )

    def reproduce(self, img_src=None, process_time=0):
        """
            If video is not paused get next video capture frame and draw gui
            components  
            Args:
            returns:
        """

        # If video is not paused get next video capture frame and draw gui
        # components
        self.tick = time.time()

        if not self._win_pause:
            self._get_frame()

            # Resize video frame if too big
            self._win_img = self.cap_img.copy() if img_src is None else img_src

            if (
                self._win_img.shape[1] > self._WIN_MAX_WIDTH
                or self._win_img.shape[0] > self._WIN_MAX_HEIGHT
            ):
                fw = self._WIN_MAX_WIDTH / self._win_img.shape[1]
                self._win_img = cv2.resize(
                    self._win_img,
                    (
                        int(self._win_img.shape[1] * fw),
                        int(self._win_img.shape[0] * fw),
                    ),
                    self._RESIZE_FLAG,
                )

            self._draw_gui(self._win_img)

        if self._win_show_menu:
            self._win_img = self._win_menu

        tock = time.time() - self.tick
        twait = (1.0 / self._cap_rate - tock - process_time) * 1000
        if twait <= 1:
            twait = 1

        # Show player window and video capture frame
        cv2.imshow(self._win_name, self._win_img)
        self._key_action(key=cv2.waitKey(int(twait)))


# =============================================================================
def main(argv):

    fwd = os.path.dirname(os.path.abspath(__file__))
    fwd = os.path.abspath(os.path.join(fwd, os.pardir))

    # Create player
    media_player = Player(
        video_src_file=os.path.join(fwd, "media", "data_src.yaml"),
        win_name="media_player",
        media_loop=True,
    )

    # start player
    while True:
        media_player.reproduce()


# =============================================================================
if __name__ == "__main__":
    main(sys.argv[1:])
