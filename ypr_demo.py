import cv2
import numpy as np
import math
import dlib

PREDICTOR_PATH = "/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
cascade_path='haarcascades/haarcascade_frontalface_default.xml'
cascade = cv2.CascadeClassifier(cascade_path)

def face_orientation(frame, landmarks):
    size = frame.shape #(height, width, color_channel)
    print(len(landmarks))
    image_points = np.array([
                            landmarks[30],     # Nose tip
                            landmarks[8],   # Chin
                            landmarks[36],     # Left eye left corner
                            landmarks[45],     # Right eye right corne
                            landmarks[48],     # Left Mouth corner
                            landmarks[54]      # Right mouth corner
                        ], dtype="double")
                        
    model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-165.0, 170.0, -135.0),     # Left eye left corner
                            (165.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner                         
                        ])

    # camera internals
    center = (size[1]/2, size[0]/2)
    focal_length = center[0] / np.tan(60/2 * np.pi / 180)
    camera_matrix = np.array(
                         [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = "double"
                         )

    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.cv2.SOLVEPNP_ITERATIVE)
  
    axis = np.float32([[500,0,0], 
                       [0,500,0], 
                       [0,0,500]])
                          
    imgpts, jac = cv2.projectPoints(axis, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    modelpts, jac2 = cv2.projectPoints(model_points, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    rvec_matrix = cv2.Rodrigues(rotation_vector)[0]
    proj_matrix = np.hstack((rvec_matrix, translation_vector))
    eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6] 
    pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]


    pitch = math.degrees(math.asin(math.sin(pitch)))
    roll = -math.degrees(math.asin(math.sin(roll)))
    yaw = math.degrees(math.asin(math.sin(yaw)))
    return yaw, pitch,roll
    
cap = cv2.VideoCapture(0)

while True : 
  re,img = cap.read() 
  image = img.copy()
  rects = cascade.detectMultiScale(img, 1.3,5)
  try :     
      x,y,w,h =rects[0]
  except IndexError:
      continue
           
  rect=dlib.rectangle(x,y,x+w,y+h)
  matrix_test = np.matrix([[p.x, p.y] for p in predictor(img, rect).parts()]) 
  landmark_list = []

  for idx, point in enumerate(matrix_test):
       pos = (point[0, 0], point[0, 1])
       landmark_list.append(pos)  


  yaw,pitch,roll = face_orientation(img,landmark_list) 
  print('pitch : ',pitch,'  roll : ',roll,' yaw : ',yaw)

  for landmark_points in landmark_list : 
    img_test = cv2.circle(image,landmark_points,1,(255,255,255),1)

  st1 = "yaw : "+str(yaw)
  st2 = "pitch : "+str(pitch)
  st3 = "roll : "+str(roll)

  cv2.putText(img_test,st1,(25,25),cv2.FONT_HERSHEY_DUPLEX,0.5,(255,0,0),1)
  cv2.putText(img_test,st2,(25,50),cv2.FONT_HERSHEY_DUPLEX,0.5,(0,255,0),1)
  cv2.putText(img_test,st3,(25,75),cv2.FONT_HERSHEY_DUPLEX,0.5,(0,0,255),1)

  cv2.imshow("display",img_test)
  if cv2.waitKey(1) == ord('q'):
        break  
cv2.destroyAllWindows()   


