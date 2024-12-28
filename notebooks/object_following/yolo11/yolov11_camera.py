import torch
import cv2
from ultralytics import YOLO


model = YOLO("./yolo11n.pt")
#path = model.export(format="onnx", dynamic=True)


cap = cv2.VideoCapture('nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1280, height=720, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv flip-method=0 ! video/x-raw, width=(int)1280, height=(int)720, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink', cv2.CAP_GSTREAMER)

#cap = cv2.VideoCapture(0)

model.overrides['verbose'] = False

# output = cv2.VideoWriter("output_video.mp4", cv2.VideoWriter_fourcc(*'mp4v'), cap.get(cv2.CAP_PROP_FPS),
#                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

while cap.isOpened():
    success, frame = cap.read()
    success, frame = cap.read()
    success, frame = cap.read()
    if not success:
        break

    results = model.predict(frame, conf=0.5)
    for result in results:
        for box in result.boxes:
            label = result.names[int(box.cls[0])]  # 提取类别名称
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # output.write(frame)
    cv2.imshow("Video Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
# output.release()
cv2.destroyAllWindows()