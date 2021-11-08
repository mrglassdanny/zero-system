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

public:
    ChessModel(CostFunction cost_fn, float learning_rate);
    ChessModel(const char *path);

    virtual Tensor *forward(Tensor *x, bool train_flg);
    virtual void backward(Tensor *pred, Tensor *y);
};