import cv2

cap = cv2.VideoCapture(3)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()
i = 1
while True:
    
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break
    # w, h  = frame.size
    # print(f'image size: {frame.shape}')
    cv2.imshow("Webcam", frame)
    key = cv2.waitKey(1)
    
    # if key == ord('s'):
    name = "/home/shasankgunturu/personal/ComputerVisionBasics/src/images/data2/captured_image_"+str(i)+".png"
    cv2.imwrite(name, frame)
    i=i+1
    print("Image saved as captured_image.jpg")
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
