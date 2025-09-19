import numpy
import pyautogui
import time
import numpy as np
import cv2 as cv
import mediapipe as mp
from charset_normalizer.md import annotations
from mediapipe.tasks.python.vision import HandLandmarkerOptions, HandLandmarkerResult
from mediapipe import solutions
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from unicodedata import category
from mediapipe.framework.formats import landmark_pb2

if __name__ == "__main__":
    Hand_Gesture = HandGestureCapture()
    Hand_Gesture.run()


class HandGestureCapture:
    def __init__(self):
        self.init_video()
        self.init_mediapipe_hands_detector()
        # 新增变量用于滑动操作
        self.finger_count = 0
        self.last_finger_count = 0
        self.finger_count_start_time = 0
        self.finger_threshold_time = 500  # 手势确认所需时间(毫秒)
        self.current_handedness = None  # 当前检测到的手（左/右）

    def init_video(self):
        self.capture = cv.VideoCapture(0)
        self.capture.set(cv.CAP_PROP_FPS, 30)
        self.capture.set(cv.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

    def init_mediapipe_hands_detector(self):
        hand_options = HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path="D:\\PythonProject2\\hand_landmarker.task"),
            running_mode=vision.RunningMode.LIVE_STREAM,
            result_callback=self.on_finish_hands
        )
        self.hand_detector = vision.HandLandmarker.create_from_options(hand_options)
        self.hand_result = None
        self.finger_index_to_angles = dict()
        self.last_trigger_cmd_timestamp = 0

    def on_finish_hands(self, result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        self.hand_result = result
        cur_time = self.get_cur_time()
        if len(self.hand_result.hand_world_landmarks) > 0:
            self.calculate_all_fingers_angles()
            print(self.finger_index_to_angles)

            # 检测左右手
            self.detect_handedness()

            # 检测手指数量
            self.detect_finger_count()

            # 根据左右手执行不同操作
            if self.current_handedness == "Right":
                self.perform_swipe_action()  # 右手执行滑动操作
            elif self.current_handedness == "Left":
                self.perform_volume_action()  # 左手执行音量操作

    def calculate_all_fingers_angles(self):
        hand_landmarks = self.hand_result.hand_world_landmarks[0]
        self.finger_index_to_angles.clear()
        for i in range(4):
            self.finger_index_to_angles[5 + i * 4] = self.calculate_fingers_angle(0, 5 + i * 4, 6 + i * 4,
                                                                                  hand_landmarks)
            self.finger_index_to_angles[6 + i * 4] = self.calculate_fingers_angle(5 + i * 4, 6 + i * 4, 7 + i * 4,
                                                                                  hand_landmarks)
            self.finger_index_to_angles[7 + i * 4] = self.calculate_fingers_angle(6 + i * 4, 7 + i * 4, 8 + i * 4,
                                                                                  hand_landmarks)

        # 计算大拇指角度
        self.finger_index_to_angles[4] = self.calculate_fingers_angle(2, 3, 4, hand_landmarks)

    def calculate_fingers_angle(self, root_index, middle_index, end_index, hand_landmarks):
        root = hand_landmarks[root_index]
        middle = hand_landmarks[middle_index]
        end = hand_landmarks[end_index]
        vec1 = np.array([root.x - middle.x, root.y - middle.y, root.z - middle.z])
        vec2 = np.array([end.x - middle.x, end.y - middle.y, end.z - middle.z])
        vec1_norm = self.normalize_vector(vec1)
        vec2_norm = self.normalize_vector(vec2)
        dot_product_result = np.dot(vec1_norm, vec2_norm)
        angle = np.rad2deg(np.arccos(dot_product_result))
        return angle

    def normalize_vector(self, vector):
        magnitude = np.linalg.norm(vector)
        if magnitude == 0:
            return vector
        return vector / magnitude

    def get_cur_time(self):
        return int(time.time() * 1000)

    def run(self):
        while 1:
            ret, frame = self.capture.read()
            if ret:
                frame_as_numpy_array = numpy.asarray(frame)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_as_numpy_array)
                self.hand_detector.detect_async(mp_image, self.get_cur_time())
                if self.hand_result:
                    frame = self.draw_landmarks_on_image(frame, self.hand_result)
                    # 在画面上显示手指数量
                    self.draw_finger_count(frame)
                    # 在画面上显示当前手的类型
                    self.draw_handedness_info(frame)
                cv.imshow("frame", frame)
                key = cv.waitKey(1)
                if key == 27:
                    break
        self.exit()

    def draw_landmarks_on_image(self, rgb_image, detection_result):
        hand_landmarks_list = detection_result.hand_landmarks
        handedness_list = detection_result.handedness
        annotated_image = np.copy(rgb_image)

        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            handedness = handedness_list[idx]

            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
            ])
            solutions.drawing_utils.draw_landmarks(
                annotated_image,
                hand_landmarks_proto,
                solutions.hands.HAND_CONNECTIONS,
                solutions.drawing_styles.get_default_hand_landmarks_style(),
                solutions.drawing_styles.get_default_hand_connections_style())

        return annotated_image

    def exit(self):
        self.capture.release()
        cv.destroyAllWindows()

    # 检测左右手
    def detect_handedness(self):
        if not self.hand_result or len(self.hand_result.handedness) == 0:
            self.current_handedness = None
            return

        # 获取手的类型（左/右）
        handedness = self.hand_result.handedness[0][0].category_name
        self.current_handedness = handedness

    # 检测伸出的手指数量
    def detect_finger_count(self):
        if not self.hand_result or len(self.hand_result.hand_landmarks) == 0:
            self.finger_count = 0
            return

        hand_landmarks = self.hand_result.hand_landmarks[0]
        finger_tip_ids = [4, 8, 12, 16, 20]  # 大拇指、食指、中指、无名指、小拇指的指尖关键点ID
        finger_pip_ids = [2, 6, 10, 14, 18]  # 大拇指、食指、中指、无名指、小拇指的近侧指间关节关键点ID

        count = 0

        # 检测大拇指
        thumb_tip = hand_landmarks[finger_tip_ids[0]]
        thumb_pip = hand_landmarks[finger_pip_ids[0]]

        # 比较大拇指指尖和近侧指间关节的x坐标（考虑左右手）
        handedness = self.current_handedness
        if handedness == "Left" and thumb_tip.x < thumb_pip.x:
            count += 1
        elif handedness == "Right" and thumb_tip.x > thumb_pip.x:
            count += 1

        # 检测其他四指
        for i in range(1, 5):
            tip = hand_landmarks[finger_tip_ids[i]]
            pip = hand_landmarks[finger_pip_ids[i]]

            # 如果指尖高于近侧指间关节，则认为手指伸出
            if tip.y < pip.y:
                count += 1

        # 平滑处理：只有当手指数量稳定一段时间后才更新
        if count != self.last_finger_count:
            self.last_finger_count = count
            self.finger_count_start_time = self.get_cur_time()
        else:
            if self.get_cur_time() - self.finger_count_start_time > self.finger_threshold_time:
                self.finger_count = count

    # 根据手指数量执行滑动操作（右手）
    def perform_swipe_action(self):
        current_time = self.get_cur_time()

        # 确保两次手势操作之间有足够的间隔
        if current_time - self.last_trigger_cmd_timestamp < 1500:
            return

        # 根据手指数量执行对应的滑动操作
        if self.finger_count == 1:
            print(f"{current_time} 上滑")
            pyautogui.press('up')  # 上滑
            self.last_trigger_cmd_timestamp = current_time
        elif self.finger_count == 2:
            print(f"{current_time} 下滑")
            pyautogui.press('down')  # 下滑
            self.last_trigger_cmd_timestamp = current_time
        elif self.finger_count == 3:
            print(f"{current_time} 左滑")
            pyautogui.press('left')  # 左滑
            self.last_trigger_cmd_timestamp = current_time
        elif self.finger_count == 4:
            print(f"{current_time} 右滑")
            pyautogui.press('right')  # 右滑
            self.last_trigger_cmd_timestamp = current_time
        elif self.finger_count == 5:
            print(f"{current_time} 空格")
            pyautogui.press('space')  # 空格
            self.last_trigger_cmd_timestamp = current_time

    # 根据手指数量执行音量操作（左手）
    def perform_volume_action(self):
        current_time = self.get_cur_time()

        # 确保两次手势操作之间有足够的间隔
        if current_time - self.last_trigger_cmd_timestamp < 1500:
            return

        # 根据手指数量执行对应的音量操作
        if self.finger_count == 5:
            print(f"{current_time} 音量增加")
            pyautogui.press('volumeup')  # 音量增加
            self.last_trigger_cmd_timestamp = current_time
        elif self.finger_count == 0:
            print(f"{current_time} 音量减少")
            pyautogui.press('volumedown')  # 音量减少
            self.last_trigger_cmd_timestamp = current_time

    # 在画面上显示手指数量
    def draw_finger_count(self, frame):
        if self.finger_count is not None:
            cv.putText(frame, f"finger: {self.finger_count}", (10, 30),
                       cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)

    # 在画面上显示当前手的类型
    def draw_handedness_info(self, frame):
        if self.current_handedness:
            text = f"hand: {self.current_handedness}"
            color = (255, 0, 0) if self.current_handedness == "Left" else (0, 0, 255)

            cv.putText(frame, text, (10, 70),
                       cv.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv.LINE_AA)


