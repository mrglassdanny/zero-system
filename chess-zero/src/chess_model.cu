#include "chess_model.cuh"

using namespace zero::core;
using namespace zero::nn;

// ChessModel functions:

ChessModel::ChessModel(CostFunction cost_fn, float learning_rate)
    : Model(cost_fn, learning_rate)
{
}

ChessModel::ChessModel(const char *path)
    : Model(path)
{
}

void ChessModel::set_piece_legality_mask(Tensor *x, bool white_mov_flg)
{
    x->to(Device::Cpu);
    reverse_one_hot_encode_board(x->get_arr(), this->board);
    get_piece_legality_mask(this->board, true, this->piece_legality_mask);
}

void ChessModel::set_move_legality_mask(Tensor *x, int piece_idx)
{
    x->to(Device::Cpu);
    reverse_one_hot_encode_board(x->get_arr(), this->board);
    get_move_legality_mask(this->board, piece_idx, this->move_legality_mask);
}

Tensor *ChessModel::predict_piece(Tensor *x)
{
    Tensor *pred = this->predict(x);

    pred->to(Device::Cpu);
    for (int i = 0; i < CHESS_BOARD_LEN; i++)
    {
        pred->get_arr()[i] *= this->piece_legality_mask[i];
    }

    return pred;
}

Tensor *ChessModel::predict_move(Tensor *x)
{
    Tensor *pred = this->predict(x);

    pred->to(Device::Cpu);
    for (int i = 0; i < CHESS_BOARD_LEN; i++)
    {
        pred->get_arr()[i] *= this->move_legality_mask[i];
    }

    return pred;
}
