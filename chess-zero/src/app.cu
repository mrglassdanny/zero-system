
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
    float model_eval;
    float minimax_eval;
};

struct MoveSearchResultTrio
{
    MoveSearchResult model_mov_res;
    MoveSearchResult minimax_mov_res;
    MoveSearchResult hybrid_mov_res;
};

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
    float flt_one_hot_board[CHESS_ONE_HOT_ENCODED_BOARD_LEN];

    float lbl;

    // Skip openings!
    int start_mov_idx = 10;

    printf("Total Games: %d\n", pgn->cnt);

    for (int game_idx = 0; game_idx < pgn->cnt; game_idx++)
    {
        PGNMoveList *pl = pgn->games[game_idx];

        // Make sure we have a winner:
        if (pl->white_won_flg || pl->black_won_flg)
        {
            // Set label now that we know who won:
            if (pl->white_won_flg)
            {
                lbl = 1.0f;
            }
            else
            {
                lbl = -1.0f;
            }

            white_mov_flg = true;

            for (int mov_idx = 0; mov_idx < pl->cnt; mov_idx++)
            {
                if (mov_idx >= start_mov_idx)
                {
                    // Make move:
                    ChessMove gm_chess_move = change_board_w_mov(board, pl->arr[mov_idx], white_mov_flg);

                    // Write board:
                    one_hot_encode_board(board, flt_one_hot_board);
                    fwrite(flt_one_hot_board, sizeof(float) * CHESS_ONE_HOT_ENCODED_BOARD_LEN, 1, boards_file);

                    // Write label:
                    fwrite(&lbl, sizeof(float), 1, labels_file);
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

    std::vector<int> x_shape{CHESS_ONE_HOT_ENCODE_COMBINATION_CNT, CHESS_BOARD_ROW_CNT, CHESS_BOARD_COL_CNT};

    OnDiskSupervisor *sup = new OnDiskSupervisor(0.95f, 0.05f, board_name_buf, label_name_buf, x_shape, 0);

    return sup;
}

void train_chess(const char *pgn_name)
{
    OnDiskSupervisor *sup = get_chess_supervisor(pgn_name);

    Model *model = new Model(CostFunction::MSE, 0.001f);

    model->add_layer(new ConvolutionalLayer(sup->get_x_shape(), 128, 1, 1, InitializationFunction::Xavier));

    model->add_layer(new ConvolutionalLayer(model->get_output_shape(), 128, 3, 3, InitializationFunction::Xavier));

    model->add_layer(new ConvolutionalLayer(model->get_output_shape(), 128, 3, 3, InitializationFunction::Xavier));

    model->add_layer(new LinearLayer(model->get_output_shape(), 2048, InitializationFunction::Xavier));
    model->add_layer(new ActivationLayer(model->get_output_shape(), ActivationFunction::Tanh));

    model->add_layer(new LinearLayer(model->get_output_shape(), 1024, InitializationFunction::Xavier));
    model->add_layer(new ActivationLayer(model->get_output_shape(), ActivationFunction::Tanh));

    model->add_layer(new LinearLayer(model->get_output_shape(), 512, InitializationFunction::Xavier));
    model->add_layer(new ActivationLayer(model->get_output_shape(), ActivationFunction::Tanh));

    model->add_layer(new LinearLayer(model->get_output_shape(), 64, InitializationFunction::Xavier));
    model->add_layer(new ActivationLayer(model->get_output_shape(), ActivationFunction::Tanh));

    model->add_layer(new LinearLayer(model->get_output_shape(), Tensor::get_cnt(sup->get_y_shape()), InitializationFunction::Xavier));
    model->add_layer(new ActivationLayer(model->get_output_shape(), ActivationFunction::Tanh));

    model->train_and_test(sup, 64, 2, "C:\\Users\\d0g0825\\Desktop\\temp\\chess-zero\\chess.csv");

    model->save("C:\\Users\\d0g0825\\Desktop\\temp\\chess-zero\\chess.nn");

    delete model;

    delete sup;
}

MoveSearchResultTrio get_best_move(int *immut_board, bool white_mov_flg, bool print_flg, int depth, Model *model)
{
    int legal_moves[CHESS_MAX_LEGAL_MOVE_CNT];
    char mov[CHESS_MAX_MOVE_LEN];

    int sim_board[CHESS_BOARD_LEN];

    float flt_one_hot_board[CHESS_ONE_HOT_ENCODED_BOARD_LEN];

    char model_mov[CHESS_MAX_MOVE_LEN];
    char minimax_mov[CHESS_MAX_MOVE_LEN];
    char hybrid_mov[CHESS_MAX_MOVE_LEN];

    float best_model_eval;
    float model_eval;
    float model_eval_tiebreaker;
    if (white_mov_flg)
    {
        best_model_eval = -1.0f;
        model_eval_tiebreaker = -1.0f;
    }
    else
    {
        best_model_eval = 1.0f;
        model_eval_tiebreaker = 1.0f;
    }

    float best_minimax_eval;
    MinimaxResult minimax_res;
    float minimax_eval_tiebreaker = -1.0f;
    if (white_mov_flg)
    {
        best_minimax_eval = -100.0f;
        minimax_eval_tiebreaker = -100.0f;
    }
    else
    {
        best_minimax_eval = 100.0f;
        minimax_eval_tiebreaker = 100.0f;
    }

    float best_hybrid_eval;
    if (white_mov_flg)
    {
        best_hybrid_eval = -100.0f;
    }
    else
    {
        best_hybrid_eval = 100.0f;
    }
    float hybrid_eval;
    float hybrid_model_eval;
    float hybrid_minimax_eval;

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

                        // Move string:
                        {
                            memset(mov, 0, CHESS_MAX_MOVE_LEN);
                            translate_srcdst_idx_to_mov(immut_board, piece_idx, legal_moves[mov_idx], mov);
                        }

                        // Model evaluation:
                        {
                            one_hot_encode_board(sim_board, flt_one_hot_board);
                            std::vector<int> x_shape{CHESS_ONE_HOT_ENCODE_COMBINATION_CNT, CHESS_BOARD_ROW_CNT, CHESS_BOARD_COL_CNT};
                            Tensor *x = new Tensor(Device::Cpu, x_shape);
                            x->set_arr(flt_one_hot_board);
                            Tensor *pred = model->predict(x);
                            model_eval = pred->get_val(0);
                            delete pred;
                            delete x;
                        }

                        // Minimax evaluation:
                        {
                            minimax_res = get_minimax(sim_board, white_mov_flg, !white_mov_flg, depth, 1, best_minimax_eval);
                        }

                        // Hybrid evaluation:
                        {
                            hybrid_eval = model_eval + activate_minimax_eval(minimax_res.eval);
                        }

                        if (print_flg)
                        {
                            printf("%s\t%f\t%f\t%d\n", mov, model_eval, minimax_res.eval, minimax_res.prune_flg);
                        }

                        if (model_eval == best_model_eval)
                        {
                            if (minimax_res.eval > minimax_eval_tiebreaker)
                            {
                                minimax_eval_tiebreaker = minimax_res.eval;
                                memcpy(model_mov, mov, CHESS_MAX_MOVE_LEN);
                            }
                        }

                        if (model_eval > best_model_eval)
                        {
                            best_model_eval = model_eval;
                            minimax_eval_tiebreaker = minimax_res.eval;
                            memcpy(model_mov, mov, CHESS_MAX_MOVE_LEN);
                        }

                        if (minimax_res.eval == best_minimax_eval)
                        {
                            if (model_eval > model_eval_tiebreaker)
                            {
                                model_eval_tiebreaker = model_eval;
                                memcpy(minimax_mov, mov, CHESS_MAX_MOVE_LEN);
                            }
                        }

                        if (minimax_res.eval > best_minimax_eval)
                        {
                            best_minimax_eval = minimax_res.eval;
                            model_eval_tiebreaker = model_eval;
                            memcpy(minimax_mov, mov, CHESS_MAX_MOVE_LEN);
                        }

                        if (hybrid_eval > best_hybrid_eval)
                        {
                            best_hybrid_eval = hybrid_eval;
                            hybrid_model_eval = model_eval;
                            hybrid_minimax_eval = minimax_res.eval;
                            memcpy(hybrid_mov, mov, CHESS_MAX_MOVE_LEN);
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

                        // Move string:
                        {
                            memset(mov, 0, CHESS_MAX_MOVE_LEN);
                            translate_srcdst_idx_to_mov(immut_board, piece_idx, legal_moves[mov_idx], mov);
                        }

                        // Model evaluation:
                        {
                            one_hot_encode_board(sim_board, flt_one_hot_board);
                            std::vector<int> x_shape{CHESS_ONE_HOT_ENCODE_COMBINATION_CNT, CHESS_BOARD_ROW_CNT, CHESS_BOARD_COL_CNT};
                            Tensor *x = new Tensor(Device::Cpu, x_shape);
                            x->set_arr(flt_one_hot_board);
                            Tensor *pred = model->predict(x);
                            model_eval = pred->get_val(0);
                            delete pred;
                            delete x;
                        }

                        // Minimax evaluation:
                        {
                            minimax_res = get_minimax(sim_board, white_mov_flg, !white_mov_flg, depth, 1, best_minimax_eval);
                        }

                        // Hybrid evaluation:
                        {
                            hybrid_eval = model_eval + activate_minimax_eval(minimax_res.eval);
                        }

                        if (print_flg)
                        {
                            printf("%s\t%f\t%f\t%d\n", mov, model_eval, minimax_res.eval, minimax_res.prune_flg);
                        }

                        if (model_eval == best_model_eval)
                        {
                            if (minimax_res.eval < minimax_eval_tiebreaker)
                            {
                                minimax_eval_tiebreaker = minimax_res.eval;
                                memcpy(model_mov, mov, CHESS_MAX_MOVE_LEN);
                            }
                        }

                        if (model_eval < best_model_eval)
                        {
                            best_model_eval = model_eval;
                            minimax_eval_tiebreaker = minimax_res.eval;
                            memcpy(model_mov, mov, CHESS_MAX_MOVE_LEN);
                        }

                        if (minimax_res.eval == best_minimax_eval)
                        {
                            if (model_eval < model_eval_tiebreaker)
                            {
                                model_eval_tiebreaker = model_eval;
                                memcpy(minimax_mov, mov, CHESS_MAX_MOVE_LEN);
                            }
                        }

                        if (minimax_res.eval < best_minimax_eval)
                        {
                            best_minimax_eval = minimax_res.eval;
                            model_eval_tiebreaker = model_eval;
                            memcpy(minimax_mov, mov, CHESS_MAX_MOVE_LEN);
                        }

                        if (hybrid_eval < best_hybrid_eval)
                        {
                            best_hybrid_eval = hybrid_eval;
                            hybrid_model_eval = model_eval;
                            hybrid_minimax_eval = minimax_res.eval;
                            memcpy(hybrid_mov, mov, CHESS_MAX_MOVE_LEN);
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

    MoveSearchResultTrio trio_mov_res;

    memcpy(trio_mov_res.model_mov_res.mov, model_mov, CHESS_MAX_MOVE_LEN);
    memcpy(trio_mov_res.minimax_mov_res.mov, minimax_mov, CHESS_MAX_MOVE_LEN);
    memcpy(trio_mov_res.hybrid_mov_res.mov, hybrid_mov, CHESS_MAX_MOVE_LEN);
    trio_mov_res.model_mov_res.model_eval = best_model_eval;
    trio_mov_res.minimax_mov_res.model_eval = model_eval_tiebreaker;
    trio_mov_res.hybrid_mov_res.model_eval = hybrid_model_eval;
    trio_mov_res.model_mov_res.minimax_eval = minimax_eval_tiebreaker;
    trio_mov_res.minimax_mov_res.minimax_eval = best_minimax_eval;
    trio_mov_res.hybrid_mov_res.minimax_eval = hybrid_minimax_eval;

    return trio_mov_res;
}

void play_chess(const char *model_path, bool white_flg, int depth, bool print_flg)
{
    Model *model = new Model(model_path);

    int *board = init_board();
    int cpy_board[CHESS_BOARD_LEN];
    char mov[CHESS_MAX_MOVE_LEN];

    bool white_mov_flg = true;

    // // Go ahead and make opening moves since we do not train the model on openings.
    {
        change_board_w_mov(board, "d4", white_mov_flg);
        white_mov_flg = !white_mov_flg;

        change_board_w_mov(board, "Nf6", white_mov_flg);
        white_mov_flg = !white_mov_flg;

        change_board_w_mov(board, "c4", white_mov_flg);
        white_mov_flg = !white_mov_flg;

        change_board_w_mov(board, "g6", white_mov_flg);
        white_mov_flg = !white_mov_flg;

        change_board_w_mov(board, "Nc3", white_mov_flg);
        white_mov_flg = !white_mov_flg;

        change_board_w_mov(board, "Bg7", white_mov_flg);
        white_mov_flg = !white_mov_flg;

        change_board_w_mov(board, "e4", white_mov_flg);
        white_mov_flg = !white_mov_flg;

        change_board_w_mov(board, "d6", white_mov_flg);
        white_mov_flg = !white_mov_flg;

        change_board_w_mov(board, "f3", white_mov_flg);
        white_mov_flg = !white_mov_flg;

        change_board_w_mov(board, "Nbd7", white_mov_flg);
        white_mov_flg = !white_mov_flg;
    }

    print_board(board);

    while (1)
    {

        printf("move\tmodel\t\tminimax\t\tpruned\n");
        printf("-------+---------------+---------------+------------\n");

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

            MoveSearchResultTrio trio_mov_res;

            copy_board(board, cpy_board);
            trio_mov_res = get_best_move(cpy_board, white_mov_flg, print_flg, depth, model);
            printf("%s\t%f\t%f\t-\n", trio_mov_res.model_mov_res.mov, trio_mov_res.model_mov_res.model_eval, trio_mov_res.model_mov_res.minimax_eval);
            printf("%s\t%f\t%f\t-\n", trio_mov_res.minimax_mov_res.mov, trio_mov_res.minimax_mov_res.model_eval, trio_mov_res.minimax_mov_res.minimax_eval);
            printf("%s\t%f\t%f\t-\n", trio_mov_res.hybrid_mov_res.mov, trio_mov_res.hybrid_mov_res.model_eval, trio_mov_res.hybrid_mov_res.minimax_eval);

            printf("-------+---------------+---------------+------------\n");

            // Now accept user input.
            memset(mov, 0, CHESS_MAX_MOVE_LEN);
            printf("WHITE (a, b, c, <custom>): ");

            std::cin >> mov;
            system("cls");

            // Allow user to confirm they want to make a recommended move.
            if (strcmp(mov, "a") == 0)
            {
                strcpy(mov, trio_mov_res.model_mov_res.mov);
            }
            else if (strcmp(mov, "b") == 0)
            {
                strcpy(mov, trio_mov_res.minimax_mov_res.mov);
            }
            else if (strcmp(mov, "c") == 0)
            {
                strcpy(mov, trio_mov_res.hybrid_mov_res.mov);
            }

            change_board_w_mov(board, mov, white_mov_flg);
            white_mov_flg = !white_mov_flg;
            print_flipped_board(board);
        }

        printf("move\tmodel\t\tminimax\t\tpruned\n");
        printf("-------+---------------+---------------+------------\n");

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

            MoveSearchResultTrio trio_mov_res;

            copy_board(board, cpy_board);
            trio_mov_res = get_best_move(cpy_board, white_mov_flg, print_flg, depth, model);
            printf("%s\t%f\t%f\t-\n", trio_mov_res.model_mov_res.mov, trio_mov_res.model_mov_res.model_eval, trio_mov_res.model_mov_res.minimax_eval);
            printf("%s\t%f\t%f\t-\n", trio_mov_res.minimax_mov_res.mov, trio_mov_res.minimax_mov_res.model_eval, trio_mov_res.minimax_mov_res.minimax_eval);
            printf("%s\t%f\t%f\t-\n", trio_mov_res.hybrid_mov_res.mov, trio_mov_res.hybrid_mov_res.model_eval, trio_mov_res.hybrid_mov_res.minimax_eval);

            printf("-------+---------------+---------------+------------\n");

            // Now accept user input.
            memset(mov, 0, CHESS_MAX_MOVE_LEN);
            printf("BLACK (a, b, c, <custom>): ");
            std::cin >> mov;
            system("cls");

            // Allow user to confirm they want to make a recommended move.
            if (strcmp(mov, "a") == 0)
            {
                strcpy(mov, trio_mov_res.model_mov_res.mov);
            }
            else if (strcmp(mov, "b") == 0)
            {
                strcpy(mov, trio_mov_res.minimax_mov_res.mov);
            }
            else if (strcmp(mov, "c") == 0)
            {
                strcpy(mov, trio_mov_res.hybrid_mov_res.mov);
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

    //dump_pgn("Capablanca");

    //train_chess("Capablanca");

    play_chess("C:\\Users\\d0g0825\\Desktop\\temp\\chess-zero\\chess.nn", true, 3, true);

    return 0;
}