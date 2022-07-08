import time
from collections import deque

import numpy as np
import cv2 as cv
import win32gui
import win32api
import win32ui
import win32con


class ScreenUtils:
    def __init__(self):
        self.screen_w = win32api.GetSystemMetrics(0)
        self.screen_h = win32api.GetSystemMetrics(1)
        self._fps_prev_time = 0

    @staticmethod
    def get_window_size(_hwnd):
        rect = win32gui.GetWindowRect(_hwnd)
        x = rect[0]
        y = rect[1]
        w = rect[2] - x
        h = rect[3] - y
        return w, h

    def get_fishing_area_rect(self):
        # fishing area is top half of screen minus a bit of space left and right
        pad_w_lr = self.screen_w // 4
        pad_h_d = self.screen_h // 2
        rect = (pad_w_lr, 0, self.screen_w - pad_w_lr * 2, pad_h_d)
        return rect

    def capture_fishing_area(self):
        # hwnd = win32gui.FindWindow(None, self.window_name)
        # w, h = self.get_window_size(hwnd)
        hwnd = None
        wDC = win32gui.GetWindowDC(hwnd)
        dcObj = win32ui.CreateDCFromHandle(wDC)
        cDC = dcObj.CreateCompatibleDC()

        fishing_area_rect = self.get_fishing_area_rect()
        x, y, w, h = fishing_area_rect

        dataBitMap = win32ui.CreateBitmap()
        dataBitMap.CreateCompatibleBitmap(dcObj, w, h)
        cDC.SelectObject(dataBitMap)
        cDC.BitBlt((0, 0), (w, h), dcObj, (x, y), win32con.SRCCOPY)
        bmpdata = dataBitMap.GetBitmapBits(True)
        screenshot = np.frombuffer(bmpdata, dtype=np.uint8)
        screenshot = screenshot.reshape(h, w, 4)

        # removes alpha channel
        screenshot = screenshot[..., :3]
        screenshot = np.ascontiguousarray(screenshot)

        # free res
        win32gui.DeleteObject(dataBitMap.GetHandle())
        dcObj.DeleteDC()
        cDC.DeleteDC()
        win32gui.ReleaseDC(hwnd, wDC)

        # show fps
        now = time.time()
        fps = 1 / (now - self._fps_prev_time)
        fps = round(fps, 2)
        # print("fps", fps)
        self._fps_prev_time = now

        return screenshot


class BiteTrigger:
    # TRIGGER_DIST_LOW = 7
    # TRIGGER_DIST_HIGH = 18
    TRIGGER_DIST_LOW = 11
    TRIGGER_DIST_HIGH = 25

    def __init__(self) -> None:
        self.points = deque(maxlen=5)
        self.dists = deque(maxlen=3)

    @staticmethod
    def get_points_distance(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    def get_average_point(self):
        if len(self.points) == 0:
            return 0, 0

        sx, sy = 0, 0
        for x, y in self.points:
            sx += x
            sy += y
        sx /= len(self.points)
        sy /= len(self.points)
        return sx, sy

    def check_dists(self):
        # if sum of dist in threshold, return True
        s_dist = sum(self.dists)
        print(s_dist)
        if s_dist > self.TRIGGER_DIST_LOW and s_dist < self.TRIGGER_DIST_HIGH:
            return True

        else:
            return False

    def add_point(self, p):
        pavg = self.get_average_point()
        dist = self.get_points_distance(p, pavg)
        self.points.append(p)
        self.dists.append(dist)


class BobberFinder:
    CAST_TIMEOUT = 23.0

    def __init__(self) -> None:
        self.cascade = cv.CascadeClassifier("dist/cascade.xml")
        self.screen = ScreenUtils()

    @staticmethod
    def draw_rects(img, rects):
        for x, y, w, h in rects:
            cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return img

    @staticmethod
    def get_rect_center(rect):
        x, y, w, h = rect
        return x + w / 2, y + h / 2

    def wait_for_bite(self):
        bite_trigger = BiteTrigger()
        start_time = time.time()

        while not bite_trigger.check_dists():
            if time.time() - start_time > self.CAST_TIMEOUT:
                raise TimeoutError("timeout")

            img = self.screen.capture_fishing_area()
            # detect a single face
            rects, reject_levels, level_weights = self.cascade.detectMultiScale3(
                img,
                minNeighbors=7,
                minSize=(50, 50),
                maxSize=(70, 70),
                outputRejectLevels=True,
            )

            if len(rects) == 0:
                continue

            # print(rects, reject_levels, level_weights)
            points = (self.get_rect_center(rect) for rect in rects)
            # keep the point with the highest weight
            point = sorted(zip(points, level_weights), key=lambda x: x[1])[-1][0]
            bite_trigger.add_point(point)

            # img = self.draw_rects(img, rects)
            # cv.imshow("img", img)
            # if cv.waitKey(1) & 0xFF == ord("q"):
            #     cv.destroyAllWindows()
            #     break

        rx, ry = bite_trigger.get_average_point()
        x, y, _, _ = self.screen.get_fishing_area_rect()
        return int(x + rx), int(y + ry)


class MouseSimulator:
    def click(self, x, y):
        win32api.SetCursorPos(
            (x, y),
        )
        time.sleep(0.2)
        win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN, x, y, 0, 0)
        time.sleep(0.2)
        win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP, x, y, 0, 0)


class KeyboardSimulator:
    def press(self, key):
        key = ord(key)
        win32api.keybd_event(key, 0, 0, 0)
        win32api.keybd_event(key, 0, win32con.KEYEVENTF_KEYUP, 0)


class FishingStateMachine:
    BOBBER_SCAN_INTERVAL = 100
    BOBBER_TRIGGER_DIST = 10

    def __init__(self, *, key_cast_pole):
        self.signaled = False
        self.state = "START"
        self.key_cast_pole = key_cast_pole
        self.mouse = MouseSimulator()
        self.keyboard = KeyboardSimulator()
        self.bobber_finder = BobberFinder()

    def s_START(self):
        return "CAST_POLE", None

    def s_CAST_POLE(self):
        self.keyboard.press(self.key_cast_pole)
        return "FIND_BOBBER", None

    def s_FIND_BOBBER(self):
        # Ensures the bobber is in the screen.
        time.sleep(1)

        try:
            bobber_point = self.bobber_finder.wait_for_bite()

        except TimeoutError:
            return "CAST_POLE", None

        else:
            return "CLICK_BOBBER", bobber_point

    def s_CLICK_BOBBER(self, x, y):
        # Ensures the bobber resting
        time.sleep(0.1)

        self.mouse.click(x, y)

        # Ensures looting
        time.sleep(2)

        return "CAST_POLE", None

    def run(self):
        pass_on_args = None

        while not self.signaled:
            callback = getattr(self, "s_{}".format(self.state))

            if not pass_on_args:
                pass_on_args = tuple()

            next_state, pass_on_args = callback(*pass_on_args)
            print("State: {} -> {}".format(self.state, next_state))
            self.state = next_state


def main():
    key_cast_pole = "1"
    fsm = FishingStateMachine(key_cast_pole=key_cast_pole)
    fsm.run()


if __name__ == "__main__":
    main()
