# app/csa_creator.py
#CSA　形式のデータをcshogiに読み込む
#CSA 仕様 http://www2.computer-shogi.org/protocol/record_v22.html
#cshogi https://tadaoyamaoka.github.io/cshogi/cshogi.html#module-cshogi

from PIL import Image
import cshogi
import pickle
from PIL import Image
import cshogi
import pickle
import cshogi
import xml.etree.ElementTree as ET

# APIテスト用CSAデータ
test_csa_data = """
V2.2
N+sente
N-gote
P1-KY-KE-GI-KI-OU-KI-GI-KE-KY
P2 *  *  *  *  *  *  *  *  * 
P3-FU-FU-FU-FU-FU-FU-FU-FU-FU
P4 *  *  *  *  *  *  *  *  * 
P5 *  *  *  *  *  *  *  *  * 
P6 *  *  *  *  *  *  *  *  * 
P7+FU+FU+FU+FU+FU+FU+FU+FU+FU
P8 *  *  *  *  *  *  *  *  * 
P9+KY+KE+GI+KI+OU+KI+GI+KE+KY
+
P+00KY00KE00KY00KE
P-00KY00KE00KY00KE
P-00AL
"""

def __main(image):
    #画像前処理機能①	yoloモデル訓練ツール、推論時に使用する
    #画像前処理機能②	pytorchモデル訓練ツール、推論時に使用する
    #推論処理機能①	yoloモデルの推論に使用
    #推論処理機能②	pytorchモデルの推論に使用
    #処理結果作成機能	推論した結果を基にＣＳＡデータを作成する
    
    #テストデータを返却
    return test_csa_data


def create_csa(image: Image.Image) -> str:  
    csa_data = __main(image)  # 仮のデータ
    return csa_data

def create_board_instance(image: Image.Image) -> bytes:
    csa_data =create_csa(image)    
    board = __parse_csa(csa_data)
    serialized_board = pickle.dumps(board)
    return serialized_board


def create_svg(image: Image.Image) -> str:
    csa_data =create_csa(image)    
    board = __parse_csa(csa_data)
    svg_data = board.to_svg()    
    return svg_data


def __test_main(csa_data):
    # テスト用
    # CSAデータを読み込む
    board = __parse_csa(csa_data)
    # SVGに変換
    svg_output = board.to_svg()
    # SVGファイルに出力
    with open('shogi_board.svg', 'w', encoding='utf-8') as f:
        f.write(svg_output)

# CSA形式のデータからboardを作成する関数
def __parse_csa(csa_data):
    parser = cshogi.Parser()
    parser.parse_csa_str(csa_data)
    #parse_csa_strでは持ち駒がうまく認識されていない
    board = cshogi.Board()
    board.set_sfen(parser.sfen)
    
    pieces = board.pieces.copy()
    pieces_in_hand_black = [0] * 7
    pieces_in_hand_white = [0] * 7

    # 持ち駒の情報を追加
    for line in csa_data.strip().split('\n'):
        if line.startswith('P+'):
            pieces_in_hand_black = __add_hand_pieces(pieces_in_hand_black, line[2:])
        elif line.startswith('P-'):
            pieces_in_hand_white = __add_hand_pieces(pieces_in_hand_white, line[2:])
    pieces_in_hand = (pieces_in_hand_black, pieces_in_hand_white)
    # 持ち駒を設定する
    board.set_pieces(pieces, pieces_in_hand)
    return board

# 持ち駒の情報を更新する関数
def __add_hand_pieces(pieces_in_hand, hand_pieces):
    piece_map = {
        'FU': 0, 'KY': 1, 'KE': 2, 
        'GI': 3, 'KI': 4, 'KA': 5, 
        'HI': 6
    }
    for i in range(0, len(hand_pieces), 4):
        count = 1 #int(hand_pieces[i:i+2])
        piece_str = hand_pieces[i+2:i+4]
        if piece_str in piece_map:
            piece = piece_map[piece_str]
            pieces_in_hand[piece] += count
    return pieces_in_hand

if __name__ == "__main__":
    #parse_csa 関数のテスト

# 持ち駒のCSAデータの例
    csa_data = test_csa_data
    __test_main(csa_data)
