#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define CHESS_BOARD_ROW_CNT 8
#define CHESS_BOARD_COL_CNT 8
#define CHESS_BOARD_LEN (CHESS_BOARD_COL_CNT * CHESS_BOARD_ROW_CNT)

#define CHESS_MAX_LEGAL_MOVE_CNT 64

#define CHESS_MAX_MOVE_LEN 8
#define CHESS_MAX_GAME_MOVE_CNT 500

#define CHESS_INVALID_VALUE -1

#define CHESS_ONE_HOT_ENCODE_COMBINATION_CNT 13
#define CHESS_ONE_HOT_ENCODED_BOARD_LEN (CHESS_BOARD_LEN * CHESS_ONE_HOT_ENCODE_COMBINATION_CNT)

typedef enum ChessPiece
{
    Empty = 0,
    WhitePawn = 1,
    WhiteKnight = 3,
    WhiteBishop = 4,
    WhiteRook = 6,
    WhiteQueen = 9,
    WhiteKing = 2,
    BlackPawn = -1,
    BlackKnight = -3,
    BlackBishop = -4,
    BlackRook = -6,
    BlackQueen = -9,
    BlackKing = -2
} ChessPiece;

typedef struct SrcDst_Idx
{
    int src_idx;
    int dst_idx;
} SrcDst_Idx;

int *init_board();

int *copy_board(int *src, int *dst);

void reset_board(int *board);

int is_piece_white(ChessPiece piece);

int is_piece_black(ChessPiece piece);

int is_piece_same_color(ChessPiece a, ChessPiece b);

int is_in_check(int *board, int white_mov_flg);

void get_legal_moves(int *board, int piece_idx, int *out, int test_in_check_flg);

SrcDst_Idx get_random_move(int *board, int white_mov_flg, int *cmp_board);

void simulate_board_change_w_srcdst_idx(int *board, int src_idx, int dst_idx, int *out);

void translate_srcdst_idx_to_mov(int *board, int src_idx, int dst_idx, char *out);

void change_board_w_mov(int *board, const char *mov, int white_mov_flg);

int boardcmp(int *a, int *b);

void print_board(int *board);

FILE *create_csv_for_boards(const char *file_name);

FILE *create_csv_for_board_labels(const char *file_name);

void dump_board_to_csv(FILE *file_ptr, int *board);

void dump_board_label_to_csv(FILE *file_ptr, float lbl);

void one_hot_encode_board(int *board, int *out);

FILE *create_csv_for_one_hot_encoded_boards(const char *file_name);

FILE *create_csv_for_one_hot_encoded_board_labels(const char *file_name);

void dump_one_hot_encoded_board_to_csv(FILE *file_ptr, int *board);

void dump_one_hot_encoded_board_label_to_csv(FILE *file_ptr, float lbl);