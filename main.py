import logging
from PIL import Image
import numpy as np
import cshogi
import pickle
import xml.etree.ElementTree as ET
import cv2
from detect_for_Shogiban_Komadai import get_shogiban_komadai
#モジュール名修正
from detect_for_pices import get_pieces_from_image
from create_csa_data import create_csa_data
import random
from kanjize import number2kanji
# ロギングの設定
logging.basicConfig(level=logging.DEBUG)

# APIテスト用CSAデータ
# 将棋盤が読み込めなかった場合に返却する。
test_csa_data = """
V2.2
N+sente
N-gote
P1-FU *  *  *  *  *  *  * -FU
P2 * -FU *  *  *  *  * -FU * 
P3 *  * -FU *  *  * -FU *  * 
P4 *  *  * -FU * -FU *  *  * 
P5 *  *  *  * +FU *  *  *  * 
P6 *  *  * +FU * +FU *  *  * 
P7 *  * +FU *  *  * +FU *  * 
P8 * +FU *  *  *  *  * +FU  * 
P9+FU *  *  *  *  *  *  * +FU
+
"""

def __main(image_array):    
    logging.debug("Entered __main function")
    try:
        # 画像チャネルを3に変換（必要な場合）
        if image_array.shape[2] == 4:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_BGRA2BGR)
        elif image_array.shape[2] == 1:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
            
        logging.debug(f"Processed image shape: {image_array.shape}")
        
        ### 1. yolo 将棋盤、駒台イメージの検出
        detection_results = get_shogiban_komadai(image_array)
        logging.debug(f"Detection results: {detection_results}")
        
        ### 2. 将棋盤データの作成
        # 将棋盤の画像にアクセス
        if 'shogiban_image' in detection_results:
            shogiban_image = detection_results['shogiban_image']
            logging.debug("Shogiban image accessed")
        else:
            logging.error("Shogiban image not found in detection results")
            return test_csa_data
        
        # yolo 将棋盤から駒イメージの検出
        detection_info_list = get_pieces_from_image(shogiban_image)
        logging.debug(f"Detection info list: {detection_info_list}")
        
        # 将棋盤 9x9の2次元リストを0で初期化
        shogi_board_state = [[0 for _ in range(9)] for _ in range(9)]
        for detection_info in detection_info_list:
            y = int(detection_info['position'] / 10)
            x = detection_info['position'] % 10
            sente_gote_flag = 1 if detection_info['class_name'] == 'sente' else -1
            shogi_board_state[y-1][x-1] = detection_info['predicted_class'] * sente_gote_flag
            logging.debug(f"Set piece at {y-1},{x-1} to {detection_info['predicted_class'] * sente_gote_flag}")
        
        ### 3. 持ち駒データの作成
        hand_pieces = [[],[]]
        if 'komadai_images' in detection_results:
            for i, komadai_image in enumerate(detection_results['komadai_images']):
                logging.debug(f"Processing komadai image {i+1}")
                # 駒台の画像処理は未実装
        
        ### 4. 最終処理結果の作成
        csa_data = create_csa_data(shogi_board_state, hand_pieces)
        logging.debug(f"CSA data created: {csa_data}")
        return csa_data
    
    except Exception as e:
        logging.error(f"Error in __main: {e}")
        return test_csa_data

def create_csa(image: Image.Image) -> str:  
    logging.debug("Entered create_csa function")
    image_array = np.array(image)
    csa_data = __main(image_array)
    return csa_data

def create_board_instance(image: Image.Image) -> bytes:
    logging.debug("Entered create_board_instance function")
    image_array = np.array(image)
    csa_data = create_csa(image_array)
    board = __parse_csa(csa_data)
    serialized_board = pickle.dumps(board)
    logging.debug("Board instance created and serialized")
    return serialized_board

def create_svg(image: Image.Image) -> str:
    logging.debug("Entered create_svg function")
    image_array = np.array(image)
    csa_data = create_csa(image_array)
    board = __parse_csa(csa_data)
    svg_data = board.to_svg()
    logging.debug("SVG data created")
    return svg_data

def create_next_move(image: Image.Image) -> str:
    #次の一手 ただしランダム
    image_array = np.array(image)
    csa_data = create_csa(image_array)
    board = __parse_csa(csa_data)
    ran = random.randrange(len(list(board.legal_moves))) # 合法手の数に合わせた乱数を生成
    lastmove = list(board.legal_moves)[ran] # 指し手を決定
    mfrom = cshogi.move_from(lastmove)
    mto = cshogi.move_to(lastmove)
    next_move= str(mfrom // 9 + 1) + number2kanji(mfrom % 9 + 1) + '→'  + str(mto // 9 + 1) + number2kanji(mto % 9 + 1)
    print(next_move)
    return next_move

def __test_main(csa_data):
    # テスト用
    board = __parse_csa(csa_data)
    svg_output = board.to_svg()
    with open('shogi_board.svg', 'w', encoding='utf-8') as f:
        f.write(svg_output)

def __parse_csa(csa_data):
    logging.debug("Entered __parse_csa function")
    parser = cshogi.Parser()
    parser.parse_csa_str(csa_data)
    board = cshogi.Board()
    board.set_sfen(parser.sfen)
    
    pieces = board.pieces.copy()
    pieces_in_hand_black = [0] * 7
    pieces_in_hand_white = [0] * 7

    for line in csa_data.strip().split('\n'):
        if line.startswith('P+'):
            pieces_in_hand_black = __add_hand_pieces(pieces_in_hand_black, line[2:])
        elif line.startswith('P-'):
            pieces_in_hand_white = __add_hand_pieces(pieces_in_hand_white, line[2:])
    pieces_in_hand = (pieces_in_hand_black, pieces_in_hand_white)
    board.set_pieces(pieces, pieces_in_hand)
    logging.debug("Parsed CSA data into board instance")
    return board

def __add_hand_pieces(pieces_in_hand, hand_pieces):
    logging.debug(f"Entered __add_hand_pieces with hand_pieces: {hand_pieces}")
    piece_map = {
        'FU': 0, 'KY': 1, 'KE': 2, 
        'GI': 3, 'KI': 4, 'KA': 5, 
        'HI': 6
    }
    for i in range(0, len(hand_pieces), 4):
        count = 1
        piece_str = hand_pieces[i+2:i+4]
        if piece_str in piece_map:
            piece = piece_map[piece_str]
            pieces_in_hand[piece] += count
            logging.debug(f"Added {count} of {piece_str} to hand pieces")
    return pieces_in_hand

if __name__ == "__main__":
    test_img_path = '/workspaces/env_shogi/input/detect_shogiban_komadai_task/スクリーンショット 2024-06-16 113533.png'
    img_data = cv2.imread(test_img_path)
    print(create_next_move(img_data))
    #print(create_svg(img_data))
