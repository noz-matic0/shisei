import cv2
import mediapipe as mp

# MediaPipeの初期化
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils  # 骨格描画のためのユーティリティ
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 動画ファイルを読み込む
video_path = 'soccer.mp4'  # 元の動画ファイルのパス
cap = cv2.VideoCapture(video_path)

# 動画の書き出し設定
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 出力フォーマット
out = cv2.VideoWriter('pose_estimation_output.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

# リサイズする幅と高さを大きめに指定
resize_width = 800  # 表示したい幅
resize_height = 450  # 表示したい高さ

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("動画の最後に到達しました。")
        break

    # BGRからRGBに変換してMediaPipeで処理
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    # 姿勢のランドマークを骨格として描画
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # フレームをリサイズ
    resized_frame = cv2.resize(frame, (resize_width, resize_height))

    # リサイズされたフレームを表示
    cv2.imshow("Pose Estimation with Body", resized_frame)

    # 骨格が描画されたフレームを動画に書き出し
    out.write(frame)

    # 'q'キーで終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 終了処理
cap.release()
out.release()
cv2.destroyAllWindows()
