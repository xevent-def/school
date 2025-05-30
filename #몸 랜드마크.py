import cv2
import mediapipe as mp
import numpy as np
from collections import deque

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

history_len = 5

# 손목, 무릎 위치 기록용 deque
left_wrist_history = deque(maxlen=history_len)
right_wrist_history = deque(maxlen=history_len)
left_knee_history = deque(maxlen=history_len)
right_knee_history = deque(maxlen=history_len)

movement_threshold = 0.01
visibility_threshold = 0.6

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

with mp_pose.Pose(min_detection_confidence=0.7,
                  min_tracking_confidence=0.7,
                  model_complexity=2) as pose:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        moved_part = ""
        moved_position = (50, 100)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )

            lm = results.pose_landmarks.landmark

            # 손목 좌표
            left_wrist = lm[mp_pose.PoseLandmark.LEFT_WRIST.value]
            right_wrist = lm[mp_pose.PoseLandmark.RIGHT_WRIST.value]

            # 무릎 좌표
            left_knee = lm[mp_pose.PoseLandmark.LEFT_KNEE.value]
            right_knee = lm[mp_pose.PoseLandmark.RIGHT_KNEE.value]

            # 신뢰도 확인 후 기록
            if left_wrist.visibility > visibility_threshold:
                left_wrist_history.append((left_wrist.x, left_wrist.y))
            if right_wrist.visibility > visibility_threshold:
                right_wrist_history.append((right_wrist.x, right_wrist.y))
            if left_knee.visibility > visibility_threshold:
                left_knee_history.append((left_knee.x, left_knee.y))
            if right_knee.visibility > visibility_threshold:
                right_knee_history.append((right_knee.x, right_knee.y))

            def compute_movement(history):
                start = np.array(history[0])
                end = np.array(history[-1])
                return np.linalg.norm(end - start)

            moves = {}

            if len(left_wrist_history) == history_len:
                moves["left_arm"] = (compute_movement(left_wrist_history), (left_wrist.x, left_wrist.y))
            if len(right_wrist_history) == history_len:
                moves["right_arm"] = (compute_movement(right_wrist_history), (right_wrist.x, right_wrist.y))
            if len(left_knee_history) == history_len:
                moves["left_leg"] = (compute_movement(left_knee_history), (left_knee.x, left_knee.y))
            if len(right_knee_history) == history_len:
                moves["right_leg"] = (compute_movement(right_knee_history), (right_knee.x, right_knee.y))

            if moves:
                moved_part_name, (max_move, pos) = max(moves.items(), key=lambda x: x[1][0])

                if max_move > movement_threshold:
                    moved_part = f"{moved_part_name} move"
                    moved_position = (int(pos[0] * image.shape[1]), int(pos[1] * image.shape[0]) - 30)

        if moved_part:
            x, y = moved_position
            x = max(10, min(x, image.shape[1] - 10))
            y = max(30, min(y, image.shape[0] - 10))
            moved_position = (x, y)

            cv2.putText(image, moved_part,
                        moved_position,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        (0, 255, 0),
                        4, cv2.LINE_AA)

        frame_resized = cv2.resize(image, (800, 600))
        cv2.imshow('Mediapipe Feed', frame_resized)

        if cv2.waitKey(10) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
