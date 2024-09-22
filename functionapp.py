import azure.functions as func
import numpy as np
import json
import chess
from tensorflow.keras.models import load_model # type: ignore

# Load the pre-trained TensorFlow model (h5 file)
model = load_model('chess_move_predictor.h5')

# Function to encode the board into a 8x8x12 tensor
def encode_board(board):
    board_tensor = np.zeros((8, 8, 12), dtype=np.float32)
    pieces = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            piece_type = pieces[piece.piece_type]
            color_offset = 0 if piece.color == chess.WHITE else 6
            board_tensor[chess.square_rank(square), chess.square_file(square), piece_type + color_offset] = 1.0
    
    return board_tensor

# Function to encode the move (from_square * 64 + to_square)
def encode_move(move):
    return move.from_square * 64 + move.to_square

# Function to get the best legal move
def get_best_legal_move(predicted_move_probs, board):
    legal_moves = list(board.legal_moves)

    legal_moves_probs = []
    for move in legal_moves:
        move_index = encode_move(move)
        legal_moves_probs.append((move, predicted_move_probs[move_index]))

    best_legal_move = max(legal_moves_probs, key=lambda x: x[1])[0]
    return best_legal_move

def main(req: func.HttpRequest) -> func.HttpResponse:
    try:
        # Get the FEN from the request body
        req_body = req.get_json()
        fen = req_body.get('fen')

        if not fen:
            return func.HttpResponse(
                "FEN not provided", 
                status_code=400
            )

        # Load the chess board from the FEN
        board = chess.Board(fen)

        # Encode the board state for the model
        input_position = np.array([encode_board(board)])

        # Predict the move probabilities from the model
        predicted_move_probs = model.predict(input_position)[0]

        # Get the best legal move
        best_move = get_best_legal_move(predicted_move_probs, board)

        # Return the best move in SAN notation
        return func.HttpResponse(
            json.dumps({"best_move": board.san(best_move)}),
            mimetype="application/json",
            status_code=200
        )
    
    except Exception as e:
        return func.HttpResponse(
            f"Error: {str(e)}", 
            status_code=500
        )
