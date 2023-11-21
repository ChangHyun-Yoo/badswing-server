import mediapipe as mp
from mediapipe.tasks import python
import math
import cv2
from ultralytics import YOLO


model_path = './static/pose_landmarker_heavy.task'
# model = YOLO('static/large.pt')
model = YOLO('static/small.pt')

nose = 0
left_ear = 7
right_ear = 8
left_shoulder = 11
right_shoulder = 12
left_elbow = 13
right_elbow = 14
left_wrist = 15
right_wrist = 16
right_index = 20
left_hip = 23
right_hip = 24
left_knee = 25
right_knee = 26
left_ankle = 27
right_ankle = 28

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO
)

options_image = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE
)


def cosine_theta(a, b, c):
    dist_a = math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)
    dist_b = math.sqrt((c.x - b.x) ** 2 + (c.y - b.y) ** 2)
    dot = (a.x - b.x) * (c.x - b.x) + (a.y - b.y) * (c.y - b.y)
    return math.acos(dot / (dist_a * dist_b))


def cosine_theta_with_converted_vector(a1, a2, b1, b2):
    dist_a = 0
    dist_b = 0
    dist_a += (a1.x - a2.x) ** 2 + (a1.y - a2.y) ** 2
    dist_b += (b1.x - b2.x) ** 2 + (b1.y - b2.y) ** 2
    dist_a = math.sqrt(dist_a)
    dist_b = math.sqrt(dist_b)

    dot = (a1.x - a2.x) * (b1.x - b2.x) + (a1.y - a2.y) * (b1.y - b2.y)
    return abs(dot / (dist_a * dist_b))


def euclidian_distance(a, b):
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


# cosine_theta 들의 곱 / (평균 거리 + 1) = 유사도로 측정
def similarity(a, b, keypoints):
    answer = 1.0
    avg = 0
    for keypoint in keypoints:
        if keypoint != nose:
            avg += euclidian_distance(a[keypoint], b[keypoint])
            answer *= cosine_theta_with_converted_vector(a[keypoint], a[nose], b[keypoint], b[nose])
    avg /= len(keypoints)
    answer /= avg + 1
    return answer


def check_direction(a, b, c):
    # a-b 벡터 기준 시계방향 180도 이내에 c-b 벡터가 있으면 -1
    return -1 if (a.x - b.x) * (c.y - b.y) - (a.y - b.y) * (c.x - b.x) < 0 else 1


with PoseLandmarker.create_from_options(options_image) as landmarker:
    mp_image_feature1 = mp.Image.create_from_file('./static/feature1.jpg')
    feature1_result = landmarker.detect(mp_image_feature1)
    mp_image_feature2 = mp.Image.create_from_file('./static/feature2.jpg')
    feature2_result = landmarker.detect(mp_image_feature2)
    mp_image_feature3 = mp.Image.create_from_file('./static/feature3.jpg')
    feature3_result = landmarker.detect(mp_image_feature3)


def get_video_landmarks(file):
    with PoseLandmarker.create_from_options(options) as landmarker:
        frames_landmark = []
        images = []
        cap = cv2.VideoCapture(file)
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        all_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        count = 0
        max_index = 0
        max = 0

        max_index2 = 0
        max2 = 0

        last_cock = {"x": 10000}
        last_cock_frames = -1
        while cap.isOpened():
            success, image = cap.read()

            if not success:
                break
            print("frame", count, " of", all_frame)
            # YOLO Detection
            yolo_results = model(image)
            cls = yolo_results[0].boxes.cls.cpu().numpy()
            conf = yolo_results[0].boxes.conf.cpu().numpy()
            xywh = yolo_results[0].boxes.xywh.cpu().numpy()

            mid_y = -1
            for i in range(len(cls)):
                # # 라켓이면
                # if(cls[i] == 1):
                #     mid_y = xywh[i][1] + xywh[i][3]

                # 셔틀콕이면
                if cls[i] == 0 and conf[i] > 0.60:
                    last_cock["x"] = xywh[i][0]
                    last_cock_frames = count

            # pose estimation
            images.append(image)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            r = landmarker.detect_for_video(mp_image, int(1000 / frame_rate) * count)
            count += 1
            try:
                frames_landmark.append(r.pose_landmarks[0])
                if r.pose_landmarks[0][right_elbow].y < r.pose_landmarks[0][right_wrist].y or r.pose_landmarks[0][right_shoulder].y > r.pose_landmarks[0][
                    right_elbow].y:
                    dist = 0
                else:
                    dist = similarity(feature1_result.pose_landmarks[0], r.pose_landmarks[0],
                                      [right_wrist, right_elbow, right_shoulder, left_wrist, left_elbow, left_shoulder])

                if dist > max:
                    max = dist
                    max_index = count

                if r.pose_landmarks[0][right_shoulder].y < r.pose_landmarks[0][right_elbow].y or r.pose_landmarks[0][nose].x < r.pose_landmarks[0][
                    right_wrist].x:
                    dist2 = 0
                else:
                    dist2 = similarity(feature2_result.pose_landmarks[0], r.pose_landmarks[0],
                                      [right_wrist, right_elbow, right_shoulder, left_wrist, left_elbow, left_shoulder])

                # if mid_y > r.pose_landmarks[0][right_shoulder].y * 1080:
                #     dist2 = 0

                if dist2 > max2:
                    max2 = dist2
                    max_index2 = count
            except:
                continue

        last_cock_frames += 1

        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_pose = mp.solutions.pose

        similarity_right = similarity(frames_landmark[max_index], feature1_result.pose_landmarks[0],
                                      [right_wrist, right_elbow, right_shoulder])
        is_left_up = (frames_landmark[max_index][left_elbow].y + frames_landmark[max_index][left_wrist].y) / 2 < frames_landmark[max_index][left_shoulder].y

        with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
            img1 = cv2.cvtColor(images[max_index], cv2.COLOR_RGB2BGR)
            img1.flags.writeable = False
            results = pose.process(img1)
            mp_drawing.draw_landmarks(
                img1,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )

        backswing_frame = cv2.cvtColor(images[max_index2], cv2.COLOR_RGB2BGR)
        hit_frame = cv2.cvtColor(images[last_cock_frames], cv2.COLOR_RGB2BGR)
        c_t = cosine_theta(frames_landmark[last_cock_frames][right_hip],
                                   frames_landmark[last_cock_frames][right_shoulder],
                                   frames_landmark[last_cock_frames][right_elbow])

        is_hit_timing = (c_t > 2.09) and (c_t < 2.97)

        after = last_cock_frames + 5
        try:
            images[after]
        except:
            after = len(images) - 1

        hit_frame_after = cv2.cvtColor(images[after], cv2.COLOR_RGB2BGR)

        # 팔꿈치가 펴졌는 지
        is_elbow_ok = cosine_theta(frames_landmark[last_cock_frames][right_shoulder],
                                   frames_landmark[last_cock_frames][right_elbow],
                                   frames_landmark[last_cock_frames][right_wrist]) > 2.62
        # 친 후에도 펴졌는 지
        is_elbow_ok_after = is_elbow_ok and cosine_theta(frames_landmark[after][right_shoulder],
                                                         frames_landmark[after][right_elbow],
                                                         frames_landmark[after][right_wrist]) > 2.62

        # 음수이면 사용된 것
        is_wrist_used = check_direction(frames_landmark[last_cock_frames][right_elbow],
                                        frames_landmark[last_cock_frames][right_wrist],
                                        frames_landmark[last_cock_frames][right_index]) * check_direction(
            frames_landmark[after][right_elbow], frames_landmark[after][right_wrist],
            frames_landmark[after][right_index])

        print(max_index, last_cock_frames, max2, is_left_up, is_hit_timing, is_elbow_ok, is_elbow_ok_after, is_wrist_used)
        return img1, backswing_frame, max2, hit_frame, hit_frame_after, similarity_right, is_left_up, last_cock_frames, is_hit_timing, is_elbow_ok, is_elbow_ok_after, is_wrist_used
