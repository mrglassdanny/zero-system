#pragma once

#include <zero_system/nn/model.cuh>

#include "chess.cuh"

using namespace zero::core;
using namespace zero::nn;

class ChessModel : public Model
{

private:
    int board[CHESS_BOARD_LEN];
    float piece_legality_mask[CHESS_BOARD_LEN];
    float move_legality_mask[CHESS_BOARD_LEN];

public:
    ChessModel(CostFunction cost_fn, float learning_rate);
    ChessModel(const char *path);

    void set_piece_legality_mask(Tensor *x, bool white_mov_flg);
    void set_move_legality_mask(Tensor *x, int piece_idx);

    Tensor *predict_piece(Tensor *x);
    Tensor *predict_move(Tensor *x);
};