 Iris Detection Using Python & OpenCV

A real-time  Iris Detection System  built using  Python ,  OpenCV , and  MediaPipe  to detect and track the iris region from live video input.  
The project demonstrates facial landmark detection, eye localization, and iris tracking — suitable for use in  biometric authentication ,  liveness detection , and  IoT-based smart security  applications.

 Features
- Real-time iris and eye landmark detection using  MediaPipe FaceMesh 
- Frame capture and visualization with  OpenCV 
- Works with both  laptop webcam  and  ESP32-CAM module 
- Lightweight and efficient — ideal for real-time use
- Can be extended for  access control  and  gaze tracking  systems



 Tech Stack
- Language: Python  
- Libraries:  OpenCV, MediaPipe  
- IDE:  PyCharm  
- Hardware (optional): Laptop webcam , logitech webcam 


 How It Works
1. The webcam captures live frames.  
2. MediaPipe detects facial landmarks and isolates eye regions.  
3. The iris position is tracked in real time and displayed using OpenCV.  
4. The detected points can be used for authentication or control systems.


 Demo
(Add a short video or image showing detection output here)

---

 Future Improvements
- Integrate with  IoT devices  for smart access control  
- Enhance accuracy under low-light conditions  
- Add  AI-based blink/liveness detection 

