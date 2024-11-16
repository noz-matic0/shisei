import mediapipe as mp
import cv2 #mediapipeとOpenCVをインポート

mp_hands=mp.solutions.hands
mp_drawing=mp.solutions.drawing_utils
mesh_drawing_spec=mp_drawing.DrawingSpec(thickness=2,color=(0,0,225))
mark_drawing_spec=mp_drawing.DrawingSpec(thickness=3,circle_radius=3,color=(0,225,0))

img_path='hand.jpg'

with mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.5,
            static_image_mode=True) as hands_detection:


    image=cv2.imread(img_path)
    image=cv2.resize(image,dsize=None,fx=0.3,fy=0.3)
    rgb_image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    height=rgb_image.shape[0]
    width=rgb_image.shape[1]

    results=hands_detection.process(rgb_image)

    annoted_image=image.copy()

    for hand_landmarks in results.multi_hand_landmarks:

        mp_drawing.draw_landmarks(
            image=annoted_image,
            landmark_list=hand_landmarks,
            connections=mp_hands.HAND_CONNECTIONS,
            landmark_drawing_spec=mark_drawing_spec,
            connection_drawing_spec=mesh_drawing_spec
        )

        cv2.imwrite('result.jpg',annoted_image)