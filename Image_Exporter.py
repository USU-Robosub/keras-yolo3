import cv2

source = cv2.VideoCapture('gate_high.mp4')
frame_count = 0
offset = 0
codec = cv2.VideoWriter_fourcc(*'DIVX')
output = cv2.VideoWriter("test.mp4", codec, 10.0, (640, 480))
frame_rate = 3
while source.isOpened():
    ret, frame = source.read()
    frame = cv2.resize(frame, (640, 480))
    if not ret: break
    frame_count += 1
    should_select_frame = frame_count % frame_rate == offset
    if not should_select_frame: continue
    file_number = round(float(frame_count) / 30, 3)
    if file_number > 14: break
    output.write(frame)
    # file_name = "data/images/test//

print("runtime: " + str(frame_count / frame_rate) + " seconds")

source.release()
output.release()
cv2.destroyAllWindows()
