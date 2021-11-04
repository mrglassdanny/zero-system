
#include <iostream>
#include <windows.h>

#include <zero_system/nn/model.cuh>

#include "chess.cuh"
#include "pgn.cuh"

using namespace zero::core;
using namespace zero::nn;

struct MoveSearchResult
{
    char mov[CHESS_MAX_MOVE_LEN];
    float depth_1_eval;
    float worst_case_eval;
};

long long get_chess_file_size(const char *name)
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
    sprintf(file_name_buf, "c:\\users\\d0g0825\\desktop\\temp\\chess-zero\\%s.bs", pgn_name);
    FILE *boards_file = fopen(file_name_buf, "wb");

    memset(file_name_buf, 0, 256);
    sprintf(file_name_buf, "c:\\users\\d0g0825\\desktop\\temp\\chess-zero\\%s.bl", pgn_name);
    FILE *labels_file = fopen(file_name_buf, "wb");

    bool white_mov_flg;

    int *board = init_board();
    int orig_board[CHESS_BOARD_LEN];
    int sim_board[CHESS_BOARD_LEN];
    int influence_board[CHESS_BOARD_LEN];
    int sim_influence_board[CHESS_BOARD_LEN];
    float flt_board[CHESS_BOARD_LEN];

    // Skip openings!
    int start_mov_idx = 10;

    printf("Total Games: %d\n", pgn->cnt);

    for (int game_idx = 0; game_idx < pgn->cnt; game_idx++)
    {
        PGNMoveList *pl = pgn->games[game_idx];

        if (pl->white_won_flg || pl->black_won_flg)
        {
            white_mov_flg = true;

            for (int mov_idx = 0; mov_idx < pl->cnt; mov_idx++)
            {
                if (mov_idx >= start_mov_idx)
                {
                    {
                        // Save original for simulations later.
                        copy_board(board, orig_board);

                        // Make move.
                        change_board_w_mov(board, pl->arr[mov_idx], white_mov_flg);

                        // What they did do:
                        {
                            // Write board state.
                            board_to_float(board, flt_board, true);
                            fwrite(flt_board, sizeof(float) * CHESS_BOARD_LEN, 1, boards_file);

                            // Get and write influence state.
                            get_influence_board(board, influence_board);
                            board_to_float(influence_board, flt_board, true);
                            fwrite(flt_board, sizeof(float) * CHESS_BOARD_LEN, 1, boards_file);

                            float label;
                            if (pl->white_won_flg)
                            {
                                label = 1.0f;
                                label *= (((float)mov_idx) / ((float)pl->cnt));
                            }
                            else if (pl->black_won_flg)
                            {
                                label = -1.0f;
                                label *= (((float)mov_idx) / ((float)pl->cnt));
                            }

                            fwrite(&label, sizeof(float), 1, labels_file);
                        }

                        // Simulate their other options.
                        for (int i = 0; i < 5; i++)
                        {
                            SrcDst_Idx sdi = get_random_move(orig_board, white_mov_flg, board);

                            if (sdi.src_idx != CHESS_INVALID_VALUE)
                            {
                                // What they didnt do:
                                {
                                    simulate_board_change_w_srcdst_idx(orig_board, sdi.src_idx, sdi.dst_idx, sim_board);

                                    // Write board state.
                                    board_to_float(sim_board, flt_board, true);
                                    fwrite(flt_board, sizeof(float) * CHESS_BOARD_LEN, 1, boards_file);

                                    // Get and write influence state.
                                    get_influence_board(sim_board, sim_influence_board);
                                    board_to_float(sim_influence_board, flt_board, true);
                                    fwrite(flt_board, sizeof(float) * CHESS_BOARD_LEN, 1, boards_file);

                                    float label;
                                    if (pl->white_won_flg)
                                    {
                                        label = -1.0f;
                                        label *= (((float)mov_idx) / ((float)pl->cnt));
                                    }
                                    else if (pl->black_won_flg)
                                    {
                                        label = 1.0f;
                                        label *= (((float)mov_idx) / ((float)pl->cnt));
                                    }

                                    fwrite(&label, sizeof(float), 1, labels_file);
                                }

                                // What they did do:
                                {
                                    // Write board state.
                                    board_to_float(board, flt_board, true);
                                    fwrite(flt_board, sizeof(float) * CHESS_BOARD_LEN, 1, boards_file);

                                    // Get and write influence state.
                                    get_influence_board(board, influence_board);
                                    board_to_float(influence_board, flt_board, true);
                                    fwrite(flt_board, sizeof(float) * CHESS_BOARD_LEN, 1, boards_file);

                                    float label;
                                    if (pl->white_won_flg)
                                    {
                                        label = 1.0f;
                                        label *= (((float)mov_idx) / ((float)pl->cnt));
                                    }
                                    else if (pl->black_won_flg)
                                    {
                                        label = -1.0f;
                                        label *= (((float)mov_idx) / ((float)pl->cnt));
                                    }

                                    fwrite(&label, sizeof(float), 1, labels_file);
                                }
                            }
                            else
                            {
                                break;
                            }
                        }
                    }
                }
                else
                {
                    // Make move.
                    change_board_w_mov(board, pl->arr[mov_idx], white_mov_flg);
                }

                white_mov_flg = !white_mov_flg;
            }

            reset_board(board);
        }

        if (game_idx % 10 == 0)
        {
            printf("%d / %d (%f%%)\n", game_idx, pgn->cnt, (((game_idx * 1.0) / (pgn->cnt * 1.0) * 100.0)));
        }
    }

    free(board);

    fclose(boards_file);
    fclose(labels_file);

    PGNImport_free(pgn);

    system("cls");
}

OnDiskSupervisor *get_chess_supervisor(const char *pgn_name)
{
    char board_name_buf[256];
    char label_name_buf[256];

    memset(board_name_buf, 0, 256);
    sprintf(board_name_buf, "c:\\users\\d0g0825\\desktop\\temp\\chess-zero\\%s.bs", pgn_name);

    memset(label_name_buf, 0, 256);
    sprintf(label_name_buf, "c:\\users\\d0g0825\\desktop\\temp\\chess-zero\\%s.bl", pgn_name);

    std::vector<int> x_shape{2, CHESS_BOARD_ROW_CNT, CHESS_BOARD_COL_CNT};

    OnDiskSupervisor *sup = new OnDiskSupervisor(1.0f, 0.0f, board_name_buf, label_name_buf, x_shape, 0);

    return sup;
}

void train_chess(const char *pgn_name)
{
    OnDiskSupervisor *sup = get_chess_supervisor(pgn_name);

    Model *model = new Model(CostFunction::MSE, 0.1f);

    model->add_layer(new ConvolutionalLayer(sup->get_x_shape(), 32, 3, 3, InitializationFunction::Xavier));
    model->add_layer(new ActivationLayer(model->get_output_shape(), ActivationFunction::ReLU));

    model->add_layer(new ConvolutionalLayer(model->get_output_shape(), 32, 3, 3, InitializationFunction::Xavier));
    model->add_layer(new ActivationLayer(model->get_output_shape(), ActivationFunction::ReLU));

    model->add_layer(new LinearLayer(model->get_output_shape(), 64, InitializationFunction::Xavier));
    model->add_layer(new ActivationLayer(model->get_output_shape(), ActivationFunction::ReLU));

    model->add_layer(new LinearLayer(model->get_output_shape(), Tensor::get_cnt(sup->get_y_shape()), InitializationFunction::Xavier));
    model->add_layer(new ActivationLayer(model->get_output_shape(), ActivationFunction::Tanh));

    model->train_and_test(sup, 128, 10, "C:\\Users\\d0g0825\\Desktop\\temp\\chess-zero\\chess.csv");

    model->save("C:\\Users\\d0g0825\\Desktop\\temp\\chess-zero\\chess.nn");

    delete model;

    delete sup;
}

MoveSearchResult get_best_move(int *immut_board, bool white_mov_flg, bool print_flg, int depth, Model *model)
{
    int legal_moves[CHESS_MAX_LEGAL_MOVE_CNT];
    char mov[CHESS_MAX_MOVE_LEN];

    int sim_board[CHESS_BOARD_LEN];
    int sim_influence_board[CHESS_BOARD_LEN];
    float flt_sim_board[CHESS_BOARD_LEN];
    float flt_sim_influence_board[CHESS_BOARD_LEN];
    float flt_board_buf[CHESS_BOARD_LEN * 2];

    float *cuda_flt_board_buf;
    cudaMalloc(&cuda_flt_board_buf, sizeof(float) * CHESS_BOARD_LEN * 2);

    float best_worst_case;
    if (white_mov_flg)
    {
        best_worst_case = -FLT_MAX;
    }
    else
    {
        best_worst_case = FLT_MAX;
    }

    float cur_worst_eval;
    if (white_mov_flg)
    {
        cur_worst_eval = -FLT_MAX;
    }
    else
    {
        cur_worst_eval = FLT_MAX;
    }

    char best_mov[CHESS_MAX_MOVE_LEN];

    float depth_1_best_eval;
    if (white_mov_flg)
    {
        depth_1_best_eval = -FLT_MAX;
    }
    else
    {
        depth_1_best_eval = FLT_MAX;
    }

    if (print_flg)
    {
        printf("move\tdepth 1\t\tdepth %d\t\tpruned\n", depth);
        printf("-------+---------------+---------------+------------\n");
    }

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
                        simulate_board_change_w_srcdst_idx(immut_board, piece_idx, legal_moves[mov_idx], sim_board);
                        get_influence_board(sim_board, sim_influence_board);

                        board_to_float(sim_board, flt_sim_board, true);
                        board_to_float(sim_influence_board, flt_sim_influence_board, true);

                        memcpy(flt_board_buf, flt_sim_board, sizeof(float) * CHESS_BOARD_LEN);
                        memcpy(&flt_board_buf[CHESS_BOARD_LEN], flt_sim_influence_board, sizeof(float) * CHESS_BOARD_LEN);

                        Tensor *x = new Tensor(Device::Cpu, 2, CHESS_BOARD_ROW_CNT, CHESS_BOARD_COL_CNT);
                        x->set_arr(flt_board_buf);

                        Tensor *pred = model->predict(x);

                        float depth_1_eval = pred->get_val(0);

                        delete pred;

                        memset(mov, 0, CHESS_MAX_MOVE_LEN);
                        translate_srcdst_idx_to_mov(immut_board, piece_idx, legal_moves[mov_idx], mov);

                        PruneEvaluation prune_eval = get_worst_case(sim_board, white_mov_flg, white_mov_flg, depth, 1, model, cur_worst_eval, cuda_flt_board_buf);

                        if (print_flg)
                        {
                            printf("%s\t%f\t%f\t%d\n", mov, depth_1_eval, prune_eval.eval, prune_eval.prune_flg);
                        }

                        if (prune_eval.eval == cur_worst_eval)
                        {
                            if (depth_1_eval > depth_1_best_eval)
                            {
                                depth_1_best_eval = depth_1_eval;
                                memcpy(best_mov, mov, CHESS_MAX_MOVE_LEN);
                            }
                        }

                        if (prune_eval.eval > cur_worst_eval)
                        {
                            cur_worst_eval = prune_eval.eval;
                        }

                        if (best_worst_case < prune_eval.eval)
                        {
                            best_worst_case = prune_eval.eval;
                            depth_1_best_eval = depth_1_eval;
                            memcpy(best_mov, mov, CHESS_MAX_MOVE_LEN);
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

                        simulate_board_change_w_srcdst_idx(immut_board, piece_idx, legal_moves[mov_idx], sim_board);
                        get_influence_board(sim_board, sim_influence_board);

                        board_to_float(sim_board, flt_sim_board, true);
                        board_to_float(sim_influence_board, flt_sim_influence_board, true);

                        memcpy(flt_board_buf, flt_sim_board, sizeof(float) * CHESS_BOARD_LEN);
                        memcpy(&flt_board_buf[CHESS_BOARD_LEN], flt_sim_influence_board, sizeof(float) * CHESS_BOARD_LEN);

                        Tensor *x = new Tensor(Device::Cpu, 2, CHESS_BOARD_ROW_CNT, CHESS_BOARD_COL_CNT);
                        x->set_arr(flt_board_buf);

                        Tensor *pred = model->predict(x);

                        float depth_1_eval = pred->get_val(0);

                        delete pred;

                        memset(mov, 0, CHESS_MAX_MOVE_LEN);
                        translate_srcdst_idx_to_mov(immut_board, piece_idx, legal_moves[mov_idx], mov);

                        PruneEvaluation prune_eval = get_worst_case(sim_board, white_mov_flg, white_mov_flg, depth, 1, model, cur_worst_eval, cuda_flt_board_buf);

                        if (print_flg)
                        {
                            printf("%s\t%f\t%f\t%d\n", mov, depth_1_eval, prune_eval.eval, prune_eval.prune_flg);
                        }

                        if (prune_eval.eval == cur_worst_eval)
                        {
                            if (depth_1_eval < depth_1_best_eval)
                            {
                                depth_1_best_eval = depth_1_eval;
                                memcpy(best_mov, mov, CHESS_MAX_MOVE_LEN);
                            }
                        }

                        if (prune_eval.eval < cur_worst_eval)
                        {
                            cur_worst_eval = prune_eval.eval;
                        }

                        if (best_worst_case > prune_eval.eval)
                        {
                            best_worst_case = prune_eval.eval;
                            depth_1_best_eval = depth_1_eval;
                            memcpy(best_mov, mov, CHESS_MAX_MOVE_LEN);
                        }
                    }
                }
            }
        }
    }

    if (print_flg)
    {
        printf("-------+---------------+---------------+------------\n");
    }

    cudaFree(cuda_flt_board_buf);

    MoveSearchResult mov_res;
    memcpy(mov_res.mov, best_mov, CHESS_MAX_MOVE_LEN);
    mov_res.depth_1_eval = depth_1_best_eval;
    mov_res.worst_case_eval = best_worst_case;
    return mov_res;
}

void play_chess(const char *model_path, bool white_flg, int depth, bool print_flg)
{
    Model *model = new Model(model_path);

    int *board = init_board();
    int cpy_board[CHESS_BOARD_LEN];
    char mov[CHESS_MAX_MOVE_LEN];

    bool white_mov_flg = true;

    // Go ahead and make opening moves since we do not train the model on openings.
    // {
    //     change_board_w_mov(board, "d4", white_mov_flg);
    //     white_mov_flg = !white_mov_flg;

    //     change_board_w_mov(board, "Nf6", white_mov_flg);
    //     white_mov_flg = !white_mov_flg;

    //     change_board_w_mov(board, "c4", white_mov_flg);
    //     white_mov_flg = !white_mov_flg;

    //     change_board_w_mov(board, "e6", white_mov_flg);
    //     white_mov_flg = !white_mov_flg;

    //     change_board_w_mov(board, "Nc3", white_mov_flg);
    //     white_mov_flg = !white_mov_flg;

    //     change_board_w_mov(board, "Bb4", white_mov_flg);
    //     white_mov_flg = !white_mov_flg;

    //     change_board_w_mov(board, "Qc2", white_mov_flg);
    //     white_mov_flg = !white_mov_flg;

    //     change_board_w_mov(board, "O-O", white_mov_flg);
    //     white_mov_flg = !white_mov_flg;

    //     change_board_w_mov(board, "a3", white_mov_flg);
    //     white_mov_flg = !white_mov_flg;

    //     change_board_w_mov(board, "Bxc3+", white_mov_flg);
    //     white_mov_flg = !white_mov_flg;
    // }

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

        change_board_w_mov(board, "d5", white_mov_flg);
        white_mov_flg = !white_mov_flg;

        change_board_w_mov(board, "cxd5", white_mov_flg);
        white_mov_flg = !white_mov_flg;

        change_board_w_mov(board, "exd5", white_mov_flg);
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
            //if (white_flg)
            {
                copy_board(board, cpy_board);
                mov_res = get_best_move(cpy_board, white_mov_flg, print_flg, depth, model);
                printf("BEST MOVE: %s\t(%f\t%f)\n", mov_res.mov, mov_res.depth_1_eval, mov_res.worst_case_eval);
            }

            // Now accept user input.
            memset(mov, 0, CHESS_MAX_MOVE_LEN);
            printf("ENTER MOVE (WHITE): ");

            std::cin >> mov;
            system("cls");

            //if (white_flg)
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
            //if (!white_flg)
            {
                copy_board(board, cpy_board);
                mov_res = get_best_move(cpy_board, white_mov_flg, print_flg, depth, model);
                printf("BEST MOVE: %s\t(%f\t%f)\n", mov_res.mov, mov_res.depth_1_eval, mov_res.worst_case_eval);
            }

            // Now accept user input.
            memset(mov, 0, CHESS_MAX_MOVE_LEN);
            printf("ENTER MOVE (BLACK): ");
            std::cin >> mov;
            system("cls");

            //if (!white_flg)
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
    srand(time(NULL));

    //dump_pgn("Fischer");

    //train_chess("Fischer");

    play_chess("C:\\Users\\d0g0825\\Desktop\\temp\\chess-zero\\chess.nn", true, 3, true);

    return 0;
}