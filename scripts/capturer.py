import cv2

cap = cv2.VideoCapture(3)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break
    # w, h  = frame.size
    # print(f'image size: {frame.shape}')
    cv2.imshow("Webcam", frame)
    key = cv2.waitKey(1)
    
    if key == ord('s'):
        cv2.imwrite("captured_image.jpg", frame)
        print("Image saved as captured_image.jpg")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
