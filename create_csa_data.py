piece_dict = {0: "* ",
        1: "FU", 2: "KY", 3: "KE", 4: "GI", 5: "KI", 6: "OU",  
        7: "KA", 8: "HI",  
        11: "TO", 12: "NY", 13: "NK", 14: "NG",
        17: "UM", 18: "RY"
    }


def create_csa_data(board_state, hand_pieces):
    """
    盤面状態リストと持ち駒リストを基にCSAデータを作成するメイン関数。

    Args:
        board_state (list): 9x9の盤面状態リスト
        hand_pieces (list): 2xNの持ち駒リスト

    Returns:
        str: CSAデータ
    """
    csa_data= _convert_board_state_to_csa(board_state) + _convert_hand_pieces_to_csa(hand_pieces)
    return csa_data


def _convert_board_state_to_csa(board_state):
    """
    盤面状態リストをCSA形式に変換する関数。

    Args:
        board_state (list): 9x9の盤面状態リスト

    Returns:
        list: CSA形式の盤面状態リスト
    """
    
    csa_board='V2.2\nN+sente\nN-gote\n'
    for row in range(9):
        csa_row = "P" + str(row+1)
        for clm in range(9):
            csa_row=csa_row + _piece_to_csa_str(board_state[row][clm])
        csa_board = csa_board + csa_row + '\n'
    csa_board = csa_board
    return csa_board

def _piece_to_csa_str(piece_num):
    """
    駒の値をCSA形式の文字列に変換する関数。
    Args:
        piece_num (int): 駒の値
    Returns:
        str: CSA形式の駒の文字列
    """
    sengo=''
    if piece_num<0:
        sengo='-'
        piece_num=piece_num * -1
    elif piece_num>0:
        sengo='+'
    else:
        sengo=' '
    return sengo + piece_dict.get(piece_num, " * ")


def _convert_hand_pieces_to_csa(hand_pieces):
    """
    持ち駒リストをCSA形式に変換する関数。

    Args:
        hand_pieces (list): 2xNの持ち駒リスト

    Returns:
        list: CSA形式の持ち駒リスト
    """
    csa_hand_pieces=''
    
    csa_top=['P+','P-']
    pieces_tpl = {}
    for player, pieces in enumerate(hand_pieces):
        pieces_tpl[player] = pieces

    for player in range(2):
        if len(pieces_tpl[player])>0:
            csa_row =  csa_top[player] 
            for piece in pieces_tpl[player]:
                csa_row= csa_row + '00' +  piece_dict.get(piece, " * ")
            csa_hand_pieces=csa_hand_pieces + csa_row + '\n'
    if len(csa_hand_pieces)>0:
        csa_hand_pieces='+\n' + csa_hand_pieces
    else:
        csa_hand_pieces='+\n'
    return csa_hand_pieces

if __name__=="__main__":
    test_board_state=[
        [-2,-3,-4,-5,-6,-5,-4,-3,-2],
        [0,-8,0,0,0,0,0,-7,0],
        [-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,0,0,0,0,0,0,0,0],            
        [0,0,0,0,0,0,0,0,0],            
        [0,0,0,0,0,0,0,0,0],            
        [1,1,1,1,1,1,1,1,1],
        [0,7,0,0,0,0,0,8,0],
        [2,3,4,5,6,5,4,3,2]
    ]
    test_hand_state=[
        [1,2],
        [2]
    ]
    print( create_csa_data(test_board_state,test_hand_state))
  
