import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from collections import deque

# MoveNet 입력 크기
input_size = 192

# 움직임 기록
history_len = 5
keypoint_index = {
    "left_wrist": 9,
    "right_wrist": 10,
    "left_knee": 13,
    "right_knee": 14,
}
histories = {
    part: deque(maxlen=history_len) for part in keypoint_index
}
movement_threshold = 0.01

# 모델 로드
interpreter = tflite.Interpreter(model_path="movenet_lightning.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def detect_pose(frame):
    img = cv2.resize(frame, (input_size, input_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = np.expand_dims(img, axis=0)
    img /= 255.0

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    keypoints = interpreter.get_tensor(output_details[0]['index'])
    return keypoints[0][0]  # (17, 3)

def compute_movement(history):
    if len(history) < history_len:
        return 0
    start = np.array(history[0])
    end = np.array(history[-1])
    return np.linalg.norm(end - start)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    keypoints = detect_pose(frame)
    h, w, _ = frame.shape
    moved_part = ""
    moved_position = (50, 100)

    for part, idx in keypoint_index.items():
        x, y, conf = keypoints[idx]
        if conf > 0.5:
            histories[part].append((x, y))

    moves = {}
    for part, hist in histories.items():
        if len(hist) == history_len:
            move = compute_movement(hist)
            moves[part] = (move, hist[-1])

    if moves:
        part, (amount, pos) = max(moves.items(), key=lambda x: x[1][0])
        if amount > movement_threshold:
            moved_part = f"{part} move"
            px, py = int(pos[0] * w), int(pos[1] * h)
            moved_position = (px, py - 20)

    if moved_part:
        cv2.putText(frame, moved_part, moved_position,
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("MoveNet Feed", frame)
    if cv2.waitKey(10) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
