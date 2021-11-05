#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <windows.h>

#include "chess.cuh"

typedef struct PGNMoveList
{
    int white_won_flg;
    int black_won_flg;
    char arr[CHESS_MAX_GAME_MOVE_CNT][CHESS_MAX_MOVE_LEN];
    int cnt;
} PGNMoveList;

typedef struct PGNImport
{
    PGNMoveList **games;
    int cnt;
    int cap;
    int alc;
} PGNImport;

long long get_chess_file_size(const char *name);

PGNImport *PGNImport_init(const char *pgn_file_name);

void PGNImport_free(PGNImport *pgn);