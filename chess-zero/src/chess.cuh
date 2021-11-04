#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#include <vector>

#include <zero_system/core/tensor.cuh>
#include <zero_system/nn/model.cuh>

#define CHESS_BOARD_ROW_CNT 8
#define CHESS_BOARD_COL_CNT 8
#define CHESS_BOARD_LEN (CHESS_BOARD_COL_CNT * CHESS_BOARD_ROW_CNT)

#define CHESS_MAX_LEGAL_MOVE_CNT 64

#define CHESS_MAX_MOVE_LEN 8
#define CHESS_MAX_GAME_MOVE_CNT 500

#define CHESS_INVALID_VALUE -1

#define CHESS_ONE_HOT_ENCODE_COMBINATION_CNT 13
#define CHESS_ONE_HOT_ENCODED_BOARD_LEN (CHESS_BOARD_LEN * CHESS_ONE_HOT_ENCODE_COMBINATION_CNT)

using namespace zero::core;
using namespace zero::nn;

typedef enum ChessPiece
{
    Empty = 0,
    WhitePawn = 1,
    WhiteKnight = 3,
    WhiteBishop = 4,
    WhiteRook = 6,
    WhiteQueen = 9,
    WhiteKing = 10,
    BlackPawn = -1,
    BlackKnight = -3,
    BlackBishop = -4,
    BlackRook = -6,
    BlackQueen = -9,
    BlackKing = -10
} ChessPiece;

struct SrcDst_Idx
{
    int src_idx;
    int dst_idx;
};

struct PruneEvaluation
{
    float eval;
    bool prune_flg;
};

int *init_board();

int *copy_board(int *src, int *dst);

void reset_board(int *board);

bool is_piece_white(ChessPiece piece);

bool is_piece_black(ChessPiece piece);

bool is_piece_same_color(ChessPiece a, ChessPiece b);

bool is_cell_under_attack(int *board, int cell_idx, bool white_pov_flg);

bool is_in_check(int *board, bool white_mov_flg);

bool is_in_checkmate(int *board, bool white_mov_flg);

void get_legal_moves(int *board, int piece_idx, int *out, bool test_in_check_flg);

void get_piece_influence(int *board, int piece_idx, int *out);

void get_influence_board(int *board, int *out);

SrcDst_Idx get_random_move(int *board, bool white_mov_flg, int *cmp_board);

void simulate_board_change_w_srcdst_idx(int *board, int src_idx, int dst_idx, int *out);

void translate_srcdst_idx_to_mov(int *board, int src_idx, int dst_idx, char *out);

void change_board_w_mov(int *board, const char *mov, bool white_mov_flg);

int boardcmp(int *a, int *b);

void print_board(int *board);

void print_influence_board(int *board);

float piece_to_float(ChessPiece piece);

void board_to_float(int *board, float *out, bool scale_down_flg);

void influence_board_to_float(int *influence_board, float *out, bool scale_down_flg);

void one_hot_encode_board(int *board, int *out);

float eval_board(int *board, Model *model, float *cuda_flt_board_buf);

PruneEvaluation get_worst_case(int *board, bool white_flg, bool cur_white_flg, int max_depth, int cur_depth, Model *model, float cur_worst_eval, float *cuda_flt_board_buf);
