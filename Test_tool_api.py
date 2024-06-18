import requests
import os

def test_api(endpoint_url, image_path, process_type, output_index):
    # 画像を読み込む
    with open(image_path, 'rb') as image_file:
        image_bytes = image_file.read()

    # FastAPIエンドポイントにリクエストを送信
    response = requests.post(
        endpoint_url,
        files={'file': ('image.jpg', image_bytes, 'image/jpeg')},
        data={'process_type': process_type}
    )

    # レスポンスを確認して出力
    if response.status_code == 200:
        print(f"Success! Process type: {process_type}")
        response_data = response.json()
        
        if process_type == "csa":
            output_path = os.path.join("output/API_test", f"csa{output_index}.txt")
            with open(output_path, "w") as file:
                file.write(response_data["csa_data"])
            print(f"CSA data written to {output_path}")
        
        elif process_type == "svg":
            output_path = os.path.join("output/API_test", f"svg{output_index}.svg")
            with open(output_path, "w") as file:
                file.write(response_data["svg_data"])
            print(f"SVG data written to {output_path}")
    else:
        print(f"Error! Status code: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    # テスト用のエンドポイントURLと画像ディレクトリ
    endpoint_url = "http://localhost:8000/process_shogi_image/"
    image_dir = '/workspaces/env_shogi/input/detect_shogiban_komadai_task/'

    # 出力ディレクトリを作成
    if not os.path.exists("output/API_test"):
        os.makedirs("output/API_test")

    # ディレクトリ内のすべての画像ファイルを取得
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

    # 各画像ファイルを順次処理
    for idx, image_file in enumerate(image_files, start=1):
        image_path = os.path.join(image_dir, image_file)
        
        for process_type in ["csa", "svg"]:
            test_api(endpoint_url, image_path, process_type, idx)
