
#include <iostream>
#include <windows.h>

#include <zero_system/nn/nn.cuh>

#include "chess.cuh"
#include "pgn.cuh"

using namespace zero::core;
using namespace zero::nn;

#define WHITE_NN_DUMP_PATH "C:\\Users\\d0g0825\\Desktop\\temp\\nn\\white-chess.nn"
#define BLACK_NN_DUMP_PATH "C:\\Users\\d0g0825\\Desktop\\temp\\nn\\black-chess.nn"

struct MoveSearchResult
{
    char mov[CHESS_MAX_MOVE_LEN];
    float eval;
    int worst_case;
};

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

void dump_pgn(const char *pgn_name)
{
    char file_name_buf[256];
    memset(file_name_buf, 0, 256);
    sprintf(file_name_buf, "c:\\users\\d0g0825\\ml-data\\chess-zero\\%s.pgn", pgn_name);
    PGNImport *pgn = PGNImport_init(file_name_buf);

    memset(file_name_buf, 0, 256);
    sprintf(file_name_buf, "c:\\users\\d0g0825\\desktop\\temp\\chess-zero\\white-%s.bs", pgn_name);
    FILE *white_boards_file = fopen(file_name_buf, "wb");

    memset(file_name_buf, 0, 256);
    sprintf(file_name_buf, "c:\\users\\d0g0825\\desktop\\temp\\chess-zero\\black-%s.bs", pgn_name);
    FILE *black_boards_file = fopen(file_name_buf, "wb");

    bool white_mov_flg;

    int *board = init_board();

    // Skip openings!
    int start_mov_idx = 10;

    printf("Total Games: %d\n", pgn->cnt);

    for (int game_idx = 0; game_idx < pgn->cnt; game_idx++)
    {
        PGNMoveList *pl = pgn->games[game_idx];

        white_mov_flg = true;

        for (int mov_idx = start_mov_idx; mov_idx < pl->cnt; mov_idx++)
        {
            FILE *boards_file = nullptr;

            if (white_mov_flg)
            {
                boards_file = white_boards_file;
            }
            else
            {
                boards_file = black_boards_file;
            }

            // Write pre-move board state first.
            fwrite(board, sizeof(int) * CHESS_BOARD_LEN, 1, boards_file);

            // Make optimal move.
            change_board_w_mov(board, pl->arr[mov_idx], white_mov_flg);

            // Now write post-move(optimal) board state.
            fwrite(board, sizeof(int) * CHESS_BOARD_LEN, 1, boards_file);

            white_mov_flg = !white_mov_flg;
        }

        reset_board(board);

        if (game_idx % 10 == 0)
        {
            printf("%d / %d (%f%%)\n", game_idx, pgn->cnt, (((game_idx * 1.0) / (pgn->cnt * 1.0) * 100.0)));
        }
    }

    free(board);

    fclose(white_boards_file);

    PGNImport_free(pgn);

    system("cls");
}

void train_nn(const char *pgn_name, bool white_flg)
{
    srand(time(NULL));

    char file_name_buf[256];

    FILE *boards_file;
    long long boards_file_size;

    if (white_flg)
    {
        memset(file_name_buf, 0, 256);
        sprintf(file_name_buf, "c:\\users\\d0g0825\\desktop\\temp\\chess-zero\\white-%s.bs", pgn_name);
        boards_file = fopen(file_name_buf, "rb");
        boards_file_size = get_file_size(file_name_buf);
    }
    else
    {
        memset(file_name_buf, 0, 256);
        sprintf(file_name_buf, "c:\\users\\d0g0825\\desktop\\temp\\chess-zero\\black-%s.bs", pgn_name);
        boards_file = fopen(file_name_buf, "rb");
        boards_file_size = get_file_size(file_name_buf);
    }

    int file_col_cnt = CHESS_BOARD_LEN * 2;
    int file_row_cnt = boards_file_size / (sizeof(int) * file_col_cnt);

    int cur_row[CHESS_BOARD_LEN * 2];
    int pre_mov_board[CHESS_BOARD_LEN];
    int post_mov_board[CHESS_BOARD_LEN];
    int sim_board[CHESS_BOARD_LEN];

    int oh_board[CHESS_ONE_HOT_ENCODED_BOARD_LEN];
    int stacked_oh_board[CHESS_ONE_HOT_ENCODED_BOARD_LEN * 2];

    std::vector<int> layer_cfg = {CHESS_ONE_HOT_ENCODED_BOARD_LEN * 2, 2048, 2048, 1024, 1024, 256, 64, 16, 1};
    NN *nn = new NN(layer_cfg, ReLU, ReLU, MSE, 0.01f);

    FILE *csv_file_ptr = fopen("C:\\Users\\d0g0825\\Desktop\\temp\\nn\\chess-train.csv", "w");
    NN::write_csv_header(csv_file_ptr);

    int epoch = 1;

    while (true)
    {
        Batch *batch = new Batch(true, 4);

        int rand_row_idx = rand() % file_row_cnt;
        fseek(boards_file, rand_row_idx * (sizeof(int) * (CHESS_BOARD_LEN * 2)), SEEK_SET);
        fread(cur_row, sizeof(int), CHESS_BOARD_LEN * 2, boards_file);

        memcpy(pre_mov_board, cur_row, sizeof(int) * CHESS_BOARD_LEN);
        memcpy(post_mov_board, &cur_row[CHESS_BOARD_LEN], sizeof(int) * CHESS_BOARD_LEN);

        // Optimal move.
        {
            one_hot_encode_board(pre_mov_board, oh_board);
            memcpy(stacked_oh_board, oh_board, sizeof(int) * CHESS_ONE_HOT_ENCODED_BOARD_LEN);

            one_hot_encode_board(post_mov_board, oh_board);
            memcpy(&stacked_oh_board[CHESS_ONE_HOT_ENCODED_BOARD_LEN], oh_board, sizeof(int) * CHESS_ONE_HOT_ENCODED_BOARD_LEN);

            Tensor *x = new Tensor(1, CHESS_ONE_HOT_ENCODED_BOARD_LEN * 2, Gpu, stacked_oh_board);
            Tensor *y = new Tensor(1, 1, Gpu);
            y->set_idx(0, 1.0f);

            batch->add(new Record(x, y));
        }

        // Non-optimal moves.
        {
            for (int rand_mov_idx = 0; rand_mov_idx < 3; rand_mov_idx++)
            {
                SrcDst_Idx sdi = get_random_move(pre_mov_board, white_flg, post_mov_board);
                simulate_board_change_w_srcdst_idx(pre_mov_board, sdi.src_idx, sdi.dst_idx, sim_board);

                one_hot_encode_board(pre_mov_board, oh_board);
                memcpy(stacked_oh_board, oh_board, sizeof(int) * CHESS_ONE_HOT_ENCODED_BOARD_LEN);

                one_hot_encode_board(sim_board, oh_board);
                memcpy(&stacked_oh_board[CHESS_ONE_HOT_ENCODED_BOARD_LEN], oh_board, sizeof(int) * CHESS_ONE_HOT_ENCODED_BOARD_LEN);

                Tensor *x = new Tensor(1, CHESS_ONE_HOT_ENCODED_BOARD_LEN * 2, Gpu, stacked_oh_board);
                Tensor *y = new Tensor(1, 1, Gpu);
                y->set_idx(0, 0.0f);

                batch->add(new Record(x, y));
            }
        }

        // Only train if batch has something in it.
        if (batch->get_size() > 0)
        {
            Report train_rpt = nn->train(batch);
            NN::write_to_csv(csv_file_ptr, epoch, train_rpt);
        }

        delete batch;

        epoch++;

        // Allow for manual override.
        {
            if (_kbhit())
            {
                if (_getch() == 'q')
                {
                    break;
                }
            }
        }
    }

    fclose(csv_file_ptr);
    fclose(boards_file);

    if (white_flg)
    {
        nn->dump(WHITE_NN_DUMP_PATH, CHESS_ONE_HOT_ENCODED_BOARD_LEN * 2);
    }
    else
    {
        nn->dump(BLACK_NN_DUMP_PATH, CHESS_ONE_HOT_ENCODED_BOARD_LEN * 2);
    }

    delete nn;
}

MoveSearchResult get_best_move(int *immut_board, bool white_mov_flg, bool print_flg, NN *nn)
{
    int legal_moves[CHESS_MAX_LEGAL_MOVE_CNT];
    char mov[CHESS_MAX_MOVE_LEN];

    int sim_board[CHESS_BOARD_LEN];

    int oh_board[CHESS_ONE_HOT_ENCODED_BOARD_LEN];
    int stacked_oh_board[CHESS_ONE_HOT_ENCODED_BOARD_LEN * 2];

    float best_eval = -FLT_MAX;

    int best_worst_case;
    if (white_mov_flg)
    {
        best_worst_case = -INT_MAX;
    }
    else
    {
        best_worst_case = INT_MAX;
    }

    char best_mov[CHESS_MAX_MOVE_LEN];

    for (int piece_idx = 0; piece_idx < CHESS_BOARD_LEN; piece_idx++)
    {
        if (white_mov_flg)
        {
            if (is_piece_white((ChessPiece)immut_board[piece_idx]))
            {
                get_legal_moves(immut_board, piece_idx, legal_moves, true);
                for (int mov_idx = 0; mov_idx < CHESS_MAX_LEGAL_MOVE_CNT; mov_idx++)
                {
                    if (legal_moves[mov_idx] == CHESS_INVALID_VALUE)
                    {
                        break;
                    }
                    else
                    {
                        one_hot_encode_board(immut_board, oh_board);
                        memcpy(stacked_oh_board, oh_board, sizeof(int) * CHESS_ONE_HOT_ENCODED_BOARD_LEN);

                        simulate_board_change_w_srcdst_idx(immut_board, piece_idx, legal_moves[mov_idx], sim_board);
                        int worst_case = get_worst_case(sim_board, true, false, 3, 1);

                        one_hot_encode_board(sim_board, oh_board);
                        memcpy(&stacked_oh_board[CHESS_ONE_HOT_ENCODED_BOARD_LEN], oh_board, sizeof(int) * CHESS_ONE_HOT_ENCODED_BOARD_LEN);

                        Tensor *x = new Tensor(1, CHESS_ONE_HOT_ENCODED_BOARD_LEN * 2, Gpu, stacked_oh_board);
                        Tensor *pred = nn->predict(x);

                        float eval = pred->get_idx(0);

                        memset(mov, 0, CHESS_MAX_MOVE_LEN);
                        translate_srcdst_idx_to_mov(immut_board, piece_idx, legal_moves[mov_idx], mov);

                        if (print_flg)
                        {
                            printf("MOVE: %s (%f\t%d)\n", mov, eval, worst_case);
                        }

                        delete x;
                        delete pred;

                        if (best_worst_case < worst_case)
                        {
                            best_worst_case = worst_case;
                            best_eval = eval;
                            memcpy(best_mov, mov, CHESS_MAX_MOVE_LEN);
                        }

                        if (best_worst_case == worst_case)
                        {
                            if (best_eval < eval)
                            {
                                best_eval = eval;
                                memcpy(best_mov, mov, CHESS_MAX_MOVE_LEN);
                            }
                        }
                    }
                }
            }
        }
        else
        {
            if (is_piece_black((ChessPiece)immut_board[piece_idx]))
            {
                get_legal_moves(immut_board, piece_idx, legal_moves, true);
                for (int mov_idx = 0; mov_idx < CHESS_MAX_LEGAL_MOVE_CNT; mov_idx++)
                {
                    if (legal_moves[mov_idx] == CHESS_INVALID_VALUE)
                    {
                        break;
                    }
                    else
                    {
                        one_hot_encode_board(immut_board, oh_board);
                        memcpy(stacked_oh_board, oh_board, sizeof(int) * CHESS_ONE_HOT_ENCODED_BOARD_LEN);

                        simulate_board_change_w_srcdst_idx(immut_board, piece_idx, legal_moves[mov_idx], sim_board);
                        int worst_case = get_worst_case(sim_board, false, true, 3, 1);

                        one_hot_encode_board(sim_board, oh_board);
                        memcpy(&stacked_oh_board[CHESS_ONE_HOT_ENCODED_BOARD_LEN], oh_board, sizeof(int) * CHESS_ONE_HOT_ENCODED_BOARD_LEN);

                        Tensor *x = new Tensor(1, CHESS_ONE_HOT_ENCODED_BOARD_LEN * 2, Gpu, stacked_oh_board);
                        Tensor *pred = nn->predict(x);

                        float eval = pred->get_idx(0);

                        memset(mov, 0, CHESS_MAX_MOVE_LEN);
                        translate_srcdst_idx_to_mov(immut_board, piece_idx, legal_moves[mov_idx], mov);

                        if (print_flg)
                        {
                            printf("MOVE: %s (%f\t%d)\n", mov, eval, worst_case);
                        }

                        delete x;
                        delete pred;

                        if (best_worst_case > worst_case)
                        {
                            best_worst_case = worst_case;
                            best_eval = eval;
                            memcpy(best_mov, mov, CHESS_MAX_MOVE_LEN);
                        }

                        if (best_worst_case == worst_case)
                        {
                            if (best_eval < eval)
                            {
                                best_eval = eval;
                                memcpy(best_mov, mov, CHESS_MAX_MOVE_LEN);
                            }
                        }
                    }
                }
            }
        }
    }

    MoveSearchResult mov_res;
    memcpy(mov_res.mov, best_mov, CHESS_MAX_MOVE_LEN);
    mov_res.worst_case = best_worst_case;
    mov_res.eval = best_eval;
    return mov_res;
}

void play_nn(bool white_flg)
{
    NN *nn;
    if (white_flg)
    {
        nn = new NN(WHITE_NN_DUMP_PATH);
    }
    else
    {
        nn = new NN(BLACK_NN_DUMP_PATH);
    }

    int *board = init_board();
    int cpy_board[CHESS_BOARD_LEN];
    char mov[CHESS_MAX_MOVE_LEN];

    bool white_mov_flg = true;

    // Go ahead and make opening moves since we do not train the model on openings.
    {
        change_board_w_mov(board, "d4", white_mov_flg);
        white_mov_flg = !white_mov_flg;

        change_board_w_mov(board, "Nf6", white_mov_flg);
        white_mov_flg = !white_mov_flg;

        change_board_w_mov(board, "c4", white_mov_flg);
        white_mov_flg = !white_mov_flg;

        change_board_w_mov(board, "e6", white_mov_flg);
        white_mov_flg = !white_mov_flg;

        change_board_w_mov(board, "Nc3", white_mov_flg);
        white_mov_flg = !white_mov_flg;

        change_board_w_mov(board, "Bb4", white_mov_flg);
        white_mov_flg = !white_mov_flg;

        change_board_w_mov(board, "Qc2", white_mov_flg);
        white_mov_flg = !white_mov_flg;

        change_board_w_mov(board, "O-O", white_mov_flg);
        white_mov_flg = !white_mov_flg;

        change_board_w_mov(board, "a3", white_mov_flg);
        white_mov_flg = !white_mov_flg;

        change_board_w_mov(board, "Bxc3+", white_mov_flg);
        white_mov_flg = !white_mov_flg;
    }

    print_board(board);

    while (1)
    {

        // White move:
        {

            if (is_in_checkmate(board, true))
            {
                printf("CHECKMATE!\n");
                break;
            }

            if (is_in_check(board, true))
            {
                printf("CHECK!\n");
            }

            MoveSearchResult mov_res;
            if (white_flg)
            {
                copy_board(board, cpy_board);
                mov_res = get_best_move(cpy_board, white_mov_flg, false, nn);
                printf("%s\t%f\t%d\n", mov_res.mov, mov_res.eval, mov_res.worst_case);
            }

            // Now accept user input.
            memset(mov, 0, CHESS_MAX_MOVE_LEN);
            printf("ENTER MOVE (WHITE): ");

            std::cin >> mov;
            system("cls");

            if (white_flg)
            {
                // Allow user to confirm they want to make recommended move.
                if (strlen(mov) <= 1)
                {
                    strcpy(mov, mov_res.mov);
                }
            }

            change_board_w_mov(board, mov, white_mov_flg);
            white_mov_flg = !white_mov_flg;
            print_board(board);
        }

        // Black move:
        {

            if (is_in_checkmate(board, false))
            {
                printf("CHECKMATE!\n");
                break;
            }

            if (is_in_check(board, false))
            {
                printf("CHECK!\n");
            }

            MoveSearchResult mov_res;
            if (!white_flg)
            {
                copy_board(board, cpy_board);
                mov_res = get_best_move(cpy_board, white_mov_flg, false, nn);
                printf("%s\t%f\t%d\n", mov_res.mov, mov_res.eval, mov_res.worst_case);
            }

            // Now accept user input.
            memset(mov, 0, CHESS_MAX_MOVE_LEN);
            printf("ENTER MOVE (BLACK): ");
            std::cin >> mov;
            system("cls");

            if (!white_flg)
            {
                // Allow user to confirm they want to make recommended move.
                if (strlen(mov) <= 1)
                {
                    strcpy(mov, mov_res.mov);
                }
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
    //dump_pgn("ALL");

    train_nn("ALL", true);

    //play_nn(true);

    return 0;
}