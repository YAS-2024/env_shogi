import requests
import os

def test_api(endpoint_url, image_path, process_type):
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
            output_path = os.path.join("output", "csa_data.txt")
            with open(output_path, "w") as file:
                file.write(response_data["csa_data"])
            print(f"CSA data written to {output_path}")
        
        elif process_type == "svg":
            output_path = os.path.join("output", "shogi_board.svg")
            with open(output_path, "w") as file:
                file.write(response_data["svg_data"])
            print(f"SVG data written to {output_path}")
    else:
        print(f"Error! Status code: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    # テスト用のエンドポイントURLと画像パス
    endpoint_url = "http://localhost:8000/process_shogi_image/"
    image_path = "bus.jpg"  # テスト用の画像ファイルパス とりあえずバス

    # 出力ディレクトリを作成
    if not os.path.exists("output"):
        os.makedirs("output")

    # CSAとSVGプロセスタイプでテスト
    for process_type in ["csa", "svg"]:
        test_api(endpoint_url, image_path, process_type)