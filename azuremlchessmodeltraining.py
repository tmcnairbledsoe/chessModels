import tensorflow as tf
import os
import chess
import chess.pgn
import numpy as np
import concurrent.futures
from azureml.core import Run

# Function to encode the board
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

# Function to encode the move
def encode_move(move):
    return move.from_square * 64 + move.to_square

# Function to load games from a PGN file
def load_games_from_file(file_path):
    positions = []
    moves = []
    
    with open(file_path) as pgn_file:
        while True:
            try:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break

                board = game.board()
                for move in game.mainline_moves():
                    if board.is_legal(move):
                        positions.append(encode_board(board))
                        moves.append(encode_move(move))
                        board.push(move)
            except Exception as e:
                print(f"Error parsing game: {e}")
    
    return np.array(positions), np.array(moves)

# Build the model
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(8, 8, 12)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(4096, activation='softmax') # 4096 = 64 * 64 possible move combinations
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Main training logic
def main():
    # Get the experiment run context (Azure ML)
    run = Run.get_context()

    # Path to your dataset file in Azure Blob Storage
    dataset_path = "your_blob_path/chess_games.pgn"
    
    # Load data
    positions, moves = load_games_from_file(dataset_path)
    
    # Convert to NumPy arrays
    positions = np.array(positions)
    moves = np.array(moves)

    # Build the model
    model = build_model()
    
    # Train the model
    model.fit(positions, moves, epochs=10, batch_size=64)
    
    # Log final accuracy to Azure ML
    final_accuracy = model.evaluate(positions, moves)[1]
    run.log('Final accuracy', final_accuracy)
    
    # Save the model to the outputs folder
    model.save(os.path.join(run.outputs, "chess_move_predictor.h5"))

if __name__ == "__main__":
    main()
