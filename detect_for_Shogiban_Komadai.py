# ライブラリのインポート
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # CUDAを無効にする
from ultralytics import YOLO
# PyTorchのCUDA無効化を明示的に設定
import torch
torch.backends.cudnn.enabled = False

import pickle
import cv2
from datetime import datetime

# データセットのパスを設定
images_dir = os.getcwd() + '/input/detect_shogiban_komadai_task'
# ベストモデルのパス
best_model_path = os.getcwd() + '/models/yolo_detect_shogiban_komadai_model/best.pt'  # パスを修正
best_model_path_pkl = os.getcwd() + '/models/yolo_detect_shogiban_komadai_model/best.pkl'  # パスを修正
# 推論結果出力ディレクトリの設定
output_dir = os.getcwd() + '/output/detect_shogiban_komadai_task'

# クラス名のリスト（yamlファイルに基づく）
class_names = ['komadai', 'shogiban']
# デバイスの設定
#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device("cpu")
def get_shogiban_komadai(image):
    results_dict = {class_name: [] for class_name in class_names}
    with open(best_model_path_pkl, 'rb') as f:
        predict_model = pickle.load(f)
    model = YOLO(best_model_path)
    model.model = predict_model
    # モデルのロード
    model.to(device)
    # 推論の実行
    results = model(image)

    # 推論結果の処理
    for result in results:
        boxes = result.boxes  # 予測結果からバウンディングボックスを取得
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            class_id = int(box.cls[0])
            confidence = box.conf[0]

            # クラスIDとクラス名の取得
            class_name = class_names[class_id] if class_id < len(class_names) else "Unknown"

            # 結果を辞書に追加
            results_dict[class_name].append({
                "bbox": (x1.item(), y1.item(), x2.item(), y2.item()),
                "confidence": confidence.item()
            })

    # クラスごとに検出された個数を表示
    #for class_name, detections in results_dict.items():
        #print(f"{class_name}: {len(detections)}")

    # クラスごとに最も可能性の高いものを取得
    final_results = {}
    for class_name, detections in results_dict.items():
        if detections:
            # 信頼度が最も高いものを選択
            best_detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
            final_results[class_name] = best_detections[:2]  # 最大2つまで選択

    # 画像を切り出すための関数
    def get_image_from_bbox(image, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        return image[y1:y2, x1:x2]

    # 将棋盤と駒台の画像を追加
    if 'shogiban' in final_results and final_results['shogiban']:
        shogiban_bbox = final_results['shogiban'][0]['bbox']
        shogiban_image = get_image_from_bbox(image, shogiban_bbox)
        final_results['shogiban_image'] = shogiban_image

    if 'komadai' in final_results and final_results['komadai']:
        komadai_images = [get_image_from_bbox(image, komadai['bbox']) for komadai in final_results['komadai']]
        final_results['komadai_images'] = komadai_images

    return final_results

def __save_detected_images(image, detection_results, output_dir):
    date_str = datetime.now().strftime("%Y%m%d")

    # 将棋盤の画像を保存
    if 'shogiban_image' in detection_results:
        shogiban_img = detection_results['shogiban_image']
        shogiban_count = len([f for f in os.listdir(output_dir) if f.startswith('shogiban_image')])
        shogiban_filename = f"shogiban_image_{date_str}_{shogiban_count:05d}.png"
        cv2.imwrite(os.path.join(output_dir, shogiban_filename), shogiban_img)

    # 駒台の画像を保存
    if 'komadai_images' in detection_results:
        for i, komadai_img in enumerate(detection_results['komadai_images']):
            komadai_count = len([f for f in os.listdir(output_dir) if f.startswith('komadai_image')])
            komadai_filename = f"komadai_image_{date_str}_{komadai_count:05d}.png"
            cv2.imwrite(os.path.join(output_dir, komadai_filename), komadai_img)

if __name__ == '__main__':
    # 画像ファイルを順に処理
    for img_file in os.listdir(images_dir):
        if img_file.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(images_dir, img_file)
            img_data = cv2.imread(img_path)
            detection_results = get_shogiban_komadai(img_data)
            __save_detected_images(img_data, detection_results, output_dir)
