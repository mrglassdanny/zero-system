
#include <iostream>

#include <windows.h>

#include <zero_system/nn/nn.cuh>

#include "chess.cuh"
#include "pgn.cuh"

using namespace zero::core;
using namespace zero::nn;

#define NN_DUMP_PATH "C:\\Users\\d0g0825\\Desktop\\temp\\nn\\chess.nn"

typedef struct DepthSearchResult
{
    char mov[CHESS_MAX_MOVE_LEN];
    float agg_eval;
} DepthSearchResult;

long long get_file_size(const char *name)
{

    HANDLE hFile = CreateFile((LPCSTR)name, GENERIC_READ,
                              FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_EXISTING,
                              FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE)
        return -1; // error condition, could call GetLastError to find out more

    LARGE_INTEGER size;
    if (!GetFileSizeEx(hFile, &size))
    {
        CloseHandle(hFile);
        return -1; // error condition, could call GetLastError to find out more
    }

    CloseHandle(hFile);
    return size.QuadPart;
}

void test_pgn_import(const char *pgn_file_name)
{
    PGNImport *pgn = PGNImport_init(pgn_file_name);

    int *board = init_board();
    int white_mov_flg;

    for (int i = 0; i < pgn->cnt; i++)
    {
        printf("###############################################################\n");
        printf("########################### GAME %d ############################\n", i + 1);
        printf("###############################################################\n");
        white_mov_flg = 1;
        for (int j = 0; j < pgn->games[i]->cnt; j++)
        {
            change_board_w_mov(board, pgn->games[i]->arr[j], white_mov_flg);
            printf("=============================== ===============================\n");
            if (white_mov_flg == 1)
            {
                printf("WHITE: ");
            }
            else
            {
                printf("BLACK: ");
            }
            printf("%s  (move %d)\n", pgn->games[i]->arr[j], j + 1);
            print_board(board);
            white_mov_flg = !white_mov_flg;
            _getch();
        }
        reset_board(board);
    }

    free(board);

    PGNImport_free(pgn);
}

void dump_pgn_one_hot_encoded_boards_to_bin(const char *pgn_file_name)
{
    PGNImport *pgn = PGNImport_init(pgn_file_name);

    FILE *boards_bin_file = fopen("c:\\users\\d0g0825\\desktop\\temp\\chess-zero\\one-hot-encoded-boards.ohbs", "wb");
    FILE *board_labels_bin_file = fopen("c:\\users\\d0g0825\\desktop\\temp\\chess-zero\\one-hot-encoded-board-labels.ohbls", "wb");

    int white_mov_flg;

    int *board = init_board();
    int cpy_board[CHESS_BOARD_LEN] = {0};
    int sim_board[CHESS_BOARD_LEN] = {0};
    int one_hot_encoded_board[CHESS_ONE_HOT_ENCODED_BOARD_LEN] = {0};
    float one_hot_encoded_board_flt[CHESS_ONE_HOT_ENCODED_BOARD_LEN] = {0.0};
    int legal_moves[CHESS_MAX_LEGAL_MOVE_CNT] = {0};

    int avg_book_mov_cnt = 7;

    printf("Total Games: %d\n", pgn->cnt);

    for (int i = 0; i < pgn->cnt; i++)
    {
        PGNMoveList *pl = pgn->games[i];

        if (pl->white_won_flg == 1 || pl->black_won_flg == 1)
        {
            white_mov_flg = 1;

            for (int j = avg_book_mov_cnt; j < pl->cnt; j++)
            {
                // Make copy before we make optimal move.
                copy_board(board, cpy_board);

                // Optimal move.
                change_board_w_mov(board, pl->arr[j], white_mov_flg);

                one_hot_encode_board(board, one_hot_encoded_board);
                for (int f = 0; f < CHESS_ONE_HOT_ENCODED_BOARD_LEN; f++)
                {
                    one_hot_encoded_board_flt[f] = (float)one_hot_encoded_board[f];
                }
                fwrite(one_hot_encoded_board_flt, sizeof(float) * CHESS_ONE_HOT_ENCODED_BOARD_LEN, 1, boards_bin_file);

                // Label.
                float lbl;

                if (pl->white_won_flg == 1)
                {
                    lbl = 1.0f;
                }
                else if (pl->black_won_flg == 1)
                {
                    lbl = -1.0f;
                }

                fwrite(&lbl, sizeof(float), 1, board_labels_bin_file);

                white_mov_flg = !white_mov_flg;
            }

            reset_board(board);

            if (i % 10 == 0)
            {
                printf("%d / %d (%f%%)\n", i, pgn->cnt, (((i * 1.0) / (pgn->cnt * 1.0) * 100.0)));
            }
        }
    }

    free(board);

    fclose(boards_bin_file);
    fclose(board_labels_bin_file);

    PGNImport_free(pgn);

    system("cls");
}

void train_nn_one_hot_encoded_bin()
{
    srand(time(NULL));

    FILE *boards_bin_file = fopen("c:\\users\\d0g0825\\desktop\\temp\\chess-zero\\one-hot-encoded-boards.ohbs", "rb");
    FILE *board_labels_bin_file = fopen("c:\\users\\d0g0825\\desktop\\temp\\chess-zero\\one-hot-encoded-board-labels.ohbls", "rb");

    long long boards_bin_file_size = get_file_size("c:\\users\\d0g0825\\desktop\\temp\\chess-zero\\one-hot-encoded-boards.ohbs");
    long long board_labels_bin_file_size = get_file_size("c:\\users\\d0g0825\\desktop\\temp\\chess-zero\\one-hot-encoded-board-labels.ohbls");

    int col_cnt = CHESS_ONE_HOT_ENCODED_BOARD_LEN;
    int row_cnt = boards_bin_file_size / (sizeof(float) * col_cnt);

    std::vector<int> layer_cfg = {col_cnt, 1024, 1024, 512, 512, 64, 1};
    NN *nn = new NN(layer_cfg, ReLU, Tanh, MSE, Xavier, 0.01f);

    float *data_buf = (float *)malloc(sizeof(float) * (row_cnt * col_cnt));
    fread(data_buf, sizeof(float) * (row_cnt * col_cnt), 1, boards_bin_file);

    float *lbl_buf = (float *)malloc(sizeof(float) * row_cnt);
    fread(lbl_buf, sizeof(float) * row_cnt, 1, board_labels_bin_file);

    Supervisor *sup = new Supervisor(row_cnt, col_cnt, 1, data_buf, lbl_buf, Cpu);

    free(data_buf);
    free(lbl_buf);

    sup->shuffle();

    nn->all(sup, 1000, 1000, "C:\\Users\\d0g0825\\Desktop\\temp\\nn\\chess-train.csv");

    nn->dump(NN_DUMP_PATH, col_cnt);

    delete nn;
    delete sup;

    fclose(boards_bin_file);
    fclose(board_labels_bin_file);
}

DepthSearchResult depth_search_single_one_hot_encoded(int *board, float agg_eval, int white_mov_flg, NN *nn)
{
    int legal_moves[CHESS_MAX_LEGAL_MOVE_CNT] = {0};
    char mov[CHESS_MAX_MOVE_LEN] = {0};

    int sim_board[CHESS_BOARD_LEN] = {0};
    int one_hot_encoded_board[CHESS_ONE_HOT_ENCODED_BOARD_LEN] = {0};

    float best_eval;
    char best_mov[CHESS_MAX_MOVE_LEN] = {0};

    if (white_mov_flg == 1)
    {
        best_eval = -FLT_MAX;
    }
    else
    {
        best_eval = FLT_MAX;
    }

    for (int i = 0; i < CHESS_BOARD_LEN; i++)
    {
        if (white_mov_flg == 1)
        {
            if (is_piece_white((ChessPiece)board[i]) == 1)
            {
                get_legal_moves(board, i, legal_moves, 1);
                for (int j = 0; j < CHESS_MAX_LEGAL_MOVE_CNT; j++)
                {
                    if (legal_moves[j] == CHESS_INVALID_VALUE)
                    {
                        break;
                    }
                    else
                    {
                        memset(mov, 0, CHESS_MAX_MOVE_LEN);
                        translate_srcdst_idx_to_mov(board, i, legal_moves[j], mov);
                        simulate_board_change_w_srcdst_idx(board, i, legal_moves[j], sim_board);

                        one_hot_encode_board(sim_board, one_hot_encoded_board);

                        Tensor *x = new Tensor(1, CHESS_ONE_HOT_ENCODED_BOARD_LEN, Gpu, one_hot_encoded_board);
                        Tensor *pred = nn->predict(x);

                        float eval = pred->get_idx(0);

                        delete x;
                        delete pred;

                        if (best_eval < agg_eval + eval)
                        {
                            best_eval = agg_eval + eval;
                            memcpy(best_mov, mov, CHESS_MAX_MOVE_LEN);
                        }
                    }
                }
            }
        }
        else
        {
            if (is_piece_black((ChessPiece)board[i]) == 1)
            {
                get_legal_moves(board, i, legal_moves, 1);
                for (int j = 0; j < CHESS_MAX_LEGAL_MOVE_CNT; j++)
                {
                    if (legal_moves[j] == CHESS_INVALID_VALUE)
                    {
                        break;
                    }
                    else
                    {
                        memset(mov, 0, CHESS_MAX_MOVE_LEN);
                        translate_srcdst_idx_to_mov(board, i, legal_moves[j], mov);
                        simulate_board_change_w_srcdst_idx(board, i, legal_moves[j], sim_board);

                        one_hot_encode_board(sim_board, one_hot_encoded_board);

                        Tensor *x = new Tensor(1, CHESS_ONE_HOT_ENCODED_BOARD_LEN, Gpu, one_hot_encoded_board);
                        Tensor *pred = nn->predict(x);

                        float eval = pred->get_idx(0);

                        delete x;
                        delete pred;

                        if (best_eval > agg_eval + eval)
                        {
                            best_eval = agg_eval + eval;
                            memcpy(best_mov, mov, CHESS_MAX_MOVE_LEN);
                        }
                    }
                }
            }
        }
    }

    DepthSearchResult ds_res;
    memcpy(ds_res.mov, best_mov, CHESS_MAX_MOVE_LEN);
    ds_res.agg_eval = best_eval;
    return ds_res;
}

DepthSearchResult depth_search_recursive_one_hot_encoded(int *immut_sim_board, int white_mov_flg, int white_mov_cur_flg, NN *nn, float agg_eval, int max_depth, int cur_depth, char *prev_mov)
{
    int mut_sim_board[CHESS_BOARD_LEN] = {0};
    int one_hot_encoded_board[CHESS_ONE_HOT_ENCODED_BOARD_LEN] = {0};

    int legal_moves[CHESS_MAX_LEGAL_MOVE_CNT] = {0};
    char mov[CHESS_MAX_MOVE_LEN] = {0};

    float best_eval;
    char best_mov[CHESS_MAX_MOVE_LEN] = {0};

    if (white_mov_flg == 1)
    {
        best_eval = -FLT_MAX;
    }
    else
    {
        best_eval = FLT_MAX;
    }

    if (white_mov_flg == 1)
    {
        for (int i = 0; i < CHESS_BOARD_LEN; i++)
        {
            if (is_piece_white((ChessPiece)immut_sim_board[i]) == 1)
            {
                get_legal_moves(immut_sim_board, i, legal_moves, 1);
                for (int j = 0; j < CHESS_MAX_LEGAL_MOVE_CNT; j++)
                {
                    if (legal_moves[j] == CHESS_INVALID_VALUE)
                    {
                        break;
                    }
                    else
                    {

                        memset(mov, 0, CHESS_MAX_MOVE_LEN);
                        translate_srcdst_idx_to_mov(immut_sim_board, i, legal_moves[j], mov);
                        simulate_board_change_w_srcdst_idx(immut_sim_board, i, legal_moves[j], mut_sim_board);

                        one_hot_encode_board(mut_sim_board, one_hot_encoded_board);

                        Tensor *x = new Tensor(1, CHESS_ONE_HOT_ENCODED_BOARD_LEN, Gpu, one_hot_encoded_board);
                        Tensor *pred = nn->predict(x);

                        float eval = pred->get_idx(0);

                        delete x;
                        delete pred;

                        if (cur_depth == max_depth)
                        {
                            if (best_eval < agg_eval + eval)
                            {
                                best_eval = agg_eval + eval;
                                memcpy(best_mov, mov, CHESS_MAX_MOVE_LEN);
                            }
                        }
                        else
                        {
                            DepthSearchResult blk_ds_res = depth_search_single_one_hot_encoded(mut_sim_board, agg_eval + eval, !white_mov_flg, nn);

                            if (strlen(blk_ds_res.mov) == 0)
                            {
                                if (best_eval < agg_eval + eval)
                                {
                                    best_eval = agg_eval + eval;
                                    memcpy(best_mov, mov, CHESS_MAX_MOVE_LEN);
                                    printf("NO MOVES: %s\n", prev_mov);
                                }
                            }
                            else
                            {
                                change_board_w_mov(mut_sim_board, blk_ds_res.mov, !white_mov_flg);
                                DepthSearchResult rec_ds_res = depth_search_recursive_one_hot_encoded(mut_sim_board, white_mov_flg, white_mov_flg, nn, blk_ds_res.agg_eval, max_depth, cur_depth + 1, mov);

                                if (best_eval < rec_ds_res.agg_eval)
                                {
                                    best_eval = rec_ds_res.agg_eval;
                                    memcpy(best_mov, mov, CHESS_MAX_MOVE_LEN);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    else
    {
        for (int i = 0; i < CHESS_BOARD_LEN; i++)
        {
            if (is_piece_black((ChessPiece)immut_sim_board[i]) == 1)
            {
                get_legal_moves(immut_sim_board, i, legal_moves, 1);
                for (int j = 0; j < CHESS_MAX_LEGAL_MOVE_CNT; j++)
                {
                    if (legal_moves[j] == CHESS_INVALID_VALUE)
                    {
                        break;
                    }
                    else
                    {

                        memset(mov, 0, CHESS_MAX_MOVE_LEN);
                        translate_srcdst_idx_to_mov(immut_sim_board, i, legal_moves[j], mov);
                        simulate_board_change_w_srcdst_idx(immut_sim_board, i, legal_moves[j], mut_sim_board);

                        one_hot_encode_board(mut_sim_board, one_hot_encoded_board);

                        Tensor *x = new Tensor(1, CHESS_ONE_HOT_ENCODED_BOARD_LEN, Gpu, one_hot_encoded_board);
                        Tensor *pred = nn->predict(x);

                        float eval = pred->get_idx(0);

                        delete x;
                        delete pred;

                        if (cur_depth == max_depth)
                        {
                            if (best_eval > agg_eval + eval)
                            {
                                best_eval = agg_eval + eval;
                                memcpy(best_mov, mov, CHESS_MAX_MOVE_LEN);
                            }
                        }
                        else
                        {
                            DepthSearchResult wht_ds_res = depth_search_single_one_hot_encoded(mut_sim_board, agg_eval + eval, !white_mov_flg, nn);
                            if (strlen(wht_ds_res.mov) == 0)
                            {
                                if (best_eval > agg_eval + eval)
                                {
                                    best_eval = agg_eval + eval;
                                    memcpy(best_mov, mov, CHESS_MAX_MOVE_LEN);
                                    printf("NO MOVES: %s\n", prev_mov);
                                }
                            }
                            else
                            {
                                change_board_w_mov(mut_sim_board, wht_ds_res.mov, !white_mov_flg);
                                DepthSearchResult rec_ds_res = depth_search_recursive_one_hot_encoded(mut_sim_board, white_mov_flg, white_mov_flg, nn, wht_ds_res.agg_eval, max_depth, cur_depth + 1, mov);

                                if (best_eval > rec_ds_res.agg_eval)
                                {
                                    best_eval = rec_ds_res.agg_eval;
                                    memcpy(best_mov, mov, CHESS_MAX_MOVE_LEN);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    DepthSearchResult ds_res;
    memcpy(ds_res.mov, best_mov, CHESS_MAX_MOVE_LEN);
    ds_res.agg_eval = best_eval;
    return ds_res;
}

void play_nn_depth_one_hot_encoded(int max_depth)
{
    NN *nn = new NN(NN_DUMP_PATH);

    int *board = init_board();
    int cpy_board[CHESS_BOARD_LEN] = {0};
    int sim_board[CHESS_BOARD_LEN] = {0};
    char mov[CHESS_MAX_MOVE_LEN] = {0};

    int legal_moves[CHESS_MAX_LEGAL_MOVE_CNT] = {0};

    int white_mov_flg = 1;
    while (1)
    {

        // White move:
        {
            copy_board(board, cpy_board);

            DepthSearchResult ds_res = depth_search_recursive_one_hot_encoded(cpy_board, white_mov_flg, white_mov_flg, nn, 0.0, max_depth, 0, NULL);
            printf("%s\t%f\n", ds_res.mov, ds_res.agg_eval);

            // Now accept user input.
            memset(mov, 0, CHESS_MAX_MOVE_LEN);
            printf("ENTER MOVE: ");
            std::cin >> mov;
            system("cls");

            // Allow user to confirm they want to make recommended move.
            if (strlen(mov) <= 1)
            {
                strcpy(mov, ds_res.mov);
            }

            change_board_w_mov(board, mov, white_mov_flg);
            white_mov_flg = !white_mov_flg;
            print_board(board);
        }

        // Black move:
        {

            copy_board(board, cpy_board);

            DepthSearchResult ds_res = depth_search_recursive_one_hot_encoded(cpy_board, white_mov_flg, white_mov_flg, nn, 0.0, max_depth, 0, NULL);
            printf("%s\t%f\n", ds_res.mov, ds_res.agg_eval);

            // Now accept user input.
            memset(mov, 0, CHESS_MAX_MOVE_LEN);
            printf("ENTER MOVE: ");
            std::cin >> mov;
            system("cls");

            // Allow user to confirm they want to make recommended move.
            if (strlen(mov) <= 1)
            {
                strcpy(mov, ds_res.mov);
            }

            change_board_w_mov(board, mov, white_mov_flg);
            white_mov_flg = !white_mov_flg;
            print_board(board);
        }
    }

    free(board);
}

int main(int argc, char **argv)
{
    //test_pgn_import("c:\\users\\d0g0825\\ml-data\\chess-zero\\TEST.pgn");

    //dump_pgn_one_hot_encoded_boards_to_bin("c:\\users\\d0g0825\\ml-data\\chess-zero\\ALL.pgn");

    train_nn_one_hot_encoded_bin();

    //play_nn_depth_one_hot_encoded(0);

    return 0;
}