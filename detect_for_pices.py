# ライブラリの読み込み
from ultralytics import YOLO
# PyTorchのCUDA無効化を明示的に設定
import torch
torch.backends.cudnn.enabled = False
import pickle
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # CUDAを無効にする
import cv2
from PIL import Image
from datetime import datetime
from Pieces_cassificarion import predict_pices_class_from_image_batch
# データセットのパスを設定

# ベストモデルのパス
best_model_path = os.getcwd() + '/models/yolo_detct_piece_model/best.pt'  # パスを修正
best_model_path_pkl = os.getcwd() + '/models/yolo_detct_piece_model/best.pkl'  # パスを修正

# 検証用の推論に使用するため　訓練に使用した画像ディレクトリ
images_dir =  os.getcwd() + '/input/detect_pieces_task'  # 正しいパスに変更
# 推論結果出力ディレクトリの設定
output_dir =  os.getcwd() +'/output/detect_pieces_task'

# クラス名のリスト（yamlファイルに基づく）
class_names = ['gote', 'sente']

# デバイスの設定
#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device("cpu")
def get_pieces_from_image(image):
    """
    画像を入力として、駒の分類結果、位置、画像を返却する関数
    """
    with open(best_model_path_pkl, 'rb') as f:
        predict_model = pickle.load(f)

    model=YOLO(best_model_path)
    model.model=predict_model
    # YOLOモデルの読み込み
    model.to(device)

    # 推論の実行
    results = model(image)

    # 画像の読み込み
    
    #img = cv2.imread(image)
    img = image #連携方法の修正により修正　6/17
    detections = []

    # 推論結果の表示と保存
    for result in results:
        boxes = result.boxes  # 予測結果からバウンティングボックスを取得

        # バウンティングボックスの面積の大きさの統計値を取得
        area_list = []
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # CPUに移動してからNumPy配列に変換
            area = (x2 - x1) * (y2 - y1)
            area_list.append(area)

        # バウンティングボックスの面積の大きさ順にソートする
        area_list = sorted(area_list)
        #print('面積リスト')
        #print(area_list)

        # 面積の小さい方から30%の閾値を取得
        threshold_index = int(len(area_list) * 0.3)
        if threshold_index == 0:
            threshold_index = 1  # リストが短すぎる場合の対策
        area_threshold = area_list[threshold_index]

        #print('面積閾値')
        #print(area_threshold)

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # CPUに移動してからNumPy配列に変換
            class_id = int(box.cls[0].cpu().numpy())

            # 取得したバウンティングボックスの面積の大きさが閾値×1.3以上の場合はスキップする
            if (x2 - x1) * (y2 - y1) > area_threshold * 1.3:
                #print(f"検出されたバウンティングボックスの面積: {(x2 - x1) * (y2 - y1)}")
                #print("バウンティングボックスの面積が大きすぎるためスキップします")
                continue

            # 個別の駒画像を抽出
            piece_img = img[int(y1):int(y2), int(x1):int(x2)]

            # バウンディングボックスの中心座標を計算
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            # 将棋盤のサイズ（9x9マス）
            board_size = 9

            # 画像の幅と高さ
            img_height, img_width = img.shape[:2]

            # 各マス目の幅と高さ
            cell_width = img_width / board_size
            cell_height = img_height / board_size

            # 中心座標がどのマス目にあるかを計算
            grid_x = int(center_x // cell_width) + 1
            grid_y = int(center_y // cell_height) + 1

            # 二桁の数字に変換（縦が一桁目、横が二桁目）
            position = grid_y * 10 + grid_x

            # 駒が "gote" の場合、画像を180度回転させる
            if class_names[class_id] == 'gote':
                piece_img = cv2.rotate(piece_img, cv2.ROTATE_180)

            # 駒画像をPIL形式に変換
            pil_img = Image.fromarray(cv2.cvtColor(piece_img, cv2.COLOR_BGR2RGB))  ####修正####

            # 検出情報を格納
            detection_info = {
                'class_id': class_id,
                'class_name': class_names[class_id] if class_id < len(class_names) else "Unknown",
                'position': position,
                'bbox': [x1, y1, x2, y2],
                'image': pil_img  ####修正####
            }

            # デバッグ情報の出力
            #print(f"検出されたクラスID: {class_id}")
            #print(f"検出されたクラス名: {detection_info['class_name']}")
            #print(f"検出されたバウンティングボックスの面積: {(x2 - x1) * (y2 - y1)}")
            #print(f"中心座標: ({center_x}, {center_y})")
            #print(f"マス目: {position}")

            detections.append(detection_info)
        
    # 駒画像を収集して分類予測を行う
    piece_images = [det['image'] for det in detections]  ####修正####
    class_predictions = predict_pices_class_from_image_batch(piece_images)  ####修正####

    # 分類結果をdetectionsに追加
    for det, class_pred in zip(detections, class_predictions):
        det['predicted_class'] = class_pred  ####修正####

    return detections
if __name__=='main':
    # 画像を順次処理
    count = 0
    os.makedirs(output_dir, exist_ok=True)

    for img_file in os.listdir(images_dir):
        if img_file.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(images_dir, img_file)
            detections = get_pieces_from_image(img_path)

            # 検出結果の画像を保存
            for detection in detections:
                current_time = datetime.now().strftime('%Y%m%d%H%M%S')
                output_filename = f"{detection['class_name']}_{current_time}_{count:05d}.png"
                output_path = os.path.join(output_dir, output_filename)
                cv2.imwrite(output_path, detection['image'])
                print(f"推論結果を保存しました: {output_path}")
                count += 1
