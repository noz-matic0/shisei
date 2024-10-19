import cv2
import mediapipe as mp

# MediaPipeの初期化
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 動画ファイルを読み込む
video_path = 'soccer.mp4'  # 動画ファイルのパス
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("動画の最後に到達しました。")
        break

    # BGRからRGBに変換
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False  # 書き込み不可にする
    results = pose.process(image)  # 姿勢推定を実行
    image.flags.writeable = True  # 書き込み可能に戻す
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # RGBからBGRに戻す

    # 姿勢のランドマークを描画
    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # 結果を表示
    cv2.imshow("Pose Estimation", image)

    # 'q'キーで終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 終了処理
cap.release()
cv2.destroyAllWindows()
