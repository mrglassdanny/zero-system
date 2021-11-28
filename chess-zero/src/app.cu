
#include <iostream>
#include <windows.h>

#include <zero_system/nn/model.cuh>

#include "chess.cuh"
#include "pgn.cuh"

using namespace zero::core;
using namespace zero::nn;

#define CHESS_START_MOVE_IDX 10

struct MoveSearchResult
{
    char mov[CHESS_MAX_MOVE_LEN];
    float minimax_eval;
    float model_eval;
};

struct Opening
{
    int all_board_states[CHESS_BOARD_LEN * CHESS_START_MOVE_IDX];
    int white_win_cnt;
    int black_win_cnt;
    int tie_cnt;
};

void dump_pgn(const char *pgn_name)
{
    char file_name_buf[256];
    memset(file_name_buf, 0, 256);
    sprintf(file_name_buf, "data\\%s.pgn", pgn_name);
    PGNImport *pgn = PGNImport_init(file_name_buf);

    memset(file_name_buf, 0, 256);
    sprintf(file_name_buf, "C:\\Users\\danny\\Desktop\\chess-zero\\%s.bs", pgn_name);
    FILE *boards_file = fopen(file_name_buf, "wb");

    memset(file_name_buf, 0, 256);
    sprintf(file_name_buf, "C:\\Users\\danny\\Desktop\\chess-zero\\%s.bl", pgn_name);
    FILE *labels_file = fopen(file_name_buf, "wb");

    bool white_mov_flg;

    int *board = init_board();
    float flt_one_hot_board[CHESS_ONE_HOT_ENCODED_BOARD_LEN];

    float lbl;

    printf("Total Games: %d\n", pgn->cnt);

    for (int game_idx = 0; game_idx < pgn->cnt; game_idx++)
    {
        PGNMoveList *pl = pgn->games[game_idx];

        {
            // Set label now that we know result:
            if (pl->white_won_flg)
            {
                lbl = 1.0f;
            }
            else if (pl->black_won_flg)
            {
                lbl = -1.0f;
            }
            else
            {
                lbl = 0.0f;
            }

            white_mov_flg = true;

            for (int mov_idx = 0; mov_idx < pl->cnt; mov_idx++)
            {
                if (mov_idx >= CHESS_START_MOVE_IDX)
                {
                    // Make move:
                    change_board_w_mov(board, pl->arr[mov_idx], white_mov_flg);

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

bool opening_game_cnt_sort_func(Opening a, Opening b)
{
    int a_cnt = a.white_win_cnt + a.black_win_cnt + a.tie_cnt;
    int b_cnt = b.white_win_cnt + b.black_win_cnt + b.tie_cnt;

    return a_cnt > b_cnt;
}

bool opening_white_wins_sort_func(Opening a, Opening b)
{
    int a_cnt = a.white_win_cnt;
    int b_cnt = b.white_win_cnt;

    return a_cnt > b_cnt;
}

bool opening_black_wins_sort_func(Opening a, Opening b)
{
    int a_cnt = a.black_win_cnt;
    int b_cnt = b.black_win_cnt;

    return a_cnt > b_cnt;
}

std::vector<Opening> get_pgn_openings(const char *pgn_name)
{
    char file_name_buf[256];
    memset(file_name_buf, 0, 256);
    sprintf(file_name_buf, "data\\%s.pgn", pgn_name);
    PGNImport *pgn = PGNImport_init(file_name_buf);

    std::vector<Opening> openings;
    int all_board_states[CHESS_BOARD_LEN * CHESS_START_MOVE_IDX];

    bool white_mov_flg;

    int *board = init_board();

    printf("Processing PGN openings...\n");

    for (int game_idx = 0; game_idx < pgn->cnt; game_idx++)
    {
        PGNMoveList *pl = pgn->games[game_idx];

        {
            white_mov_flg = true;

            for (int mov_idx = 0; mov_idx < CHESS_START_MOVE_IDX; mov_idx++)
            {
                // Make move:
                change_board_w_mov(board, pl->arr[mov_idx], white_mov_flg);

                memcpy(&all_board_states[mov_idx * CHESS_BOARD_LEN], board, sizeof(int) * CHESS_BOARD_LEN);

                white_mov_flg = !white_mov_flg;
            }

            bool opening_exists_flg = false;

            for (int mov_idx = 0; mov_idx < openings.size(); mov_idx++)
            {
                if (memcmp(all_board_states, openings[mov_idx].all_board_states, sizeof(int) * (CHESS_BOARD_LEN * CHESS_START_MOVE_IDX)) == 0)
                {
                    opening_exists_flg = true;

                    if (pl->white_won_flg)
                    {
                        openings[mov_idx].white_win_cnt++;
                    }
                    else if (pl->black_won_flg)
                    {
                        openings[mov_idx].black_win_cnt++;
                    }
                    else
                    {
                        openings[mov_idx].tie_cnt++;
                    }

                    break;
                }
            }

            if (!opening_exists_flg)
            {
                Opening opening;

                memcpy(opening.all_board_states, all_board_states, sizeof(int) * (CHESS_BOARD_LEN * CHESS_START_MOVE_IDX));
                opening.white_win_cnt = 0;
                opening.black_win_cnt = 0;
                opening.tie_cnt = 0;

                if (pl->white_won_flg)
                {
                    opening.white_win_cnt++;
                }
                else if (pl->black_won_flg)
                {
                    opening.black_win_cnt++;
                }
                else
                {
                    opening.tie_cnt++;
                }

                openings.push_back(opening);
            }

            reset_board(board);
        }
    }

    free(board);

    PGNImport_free(pgn);

    system("cls");

    return openings;
}

OnDiskSupervisor *get_chess_train_supervisor(const char *pgn_name)
{
    char board_name_buf[256];
    char label_name_buf[256];

    memset(board_name_buf, 0, 256);
    sprintf(board_name_buf, "C:\\Users\\danny\\Desktop\\chess-zero\\%s.bs", pgn_name);

    memset(label_name_buf, 0, 256);
    sprintf(label_name_buf, "C:\\Users\\danny\\Desktop\\chess-zero\\%s.bl", pgn_name);

    std::vector<int> x_shape{CHESS_ONE_HOT_ENCODE_COMBINATION_CNT, CHESS_BOARD_ROW_CNT, CHESS_BOARD_COL_CNT};

    OnDiskSupervisor *sup = new OnDiskSupervisor(1.00f, 0.00f, board_name_buf, label_name_buf, x_shape, 0);

    return sup;
}

OnDiskSupervisor *get_chess_test_supervisor(const char *pgn_name)
{
    char board_name_buf[256];
    char label_name_buf[256];

    memset(board_name_buf, 0, 256);
    sprintf(board_name_buf, "C:\\Users\\danny\\Desktop\\chess-zero\\%s.bs", pgn_name);

    memset(label_name_buf, 0, 256);
    sprintf(label_name_buf, "C:\\Users\\danny\\Desktop\\chess-zero\\%s.bl", pgn_name);

    std::vector<int> x_shape{CHESS_ONE_HOT_ENCODE_COMBINATION_CNT, CHESS_BOARD_ROW_CNT, CHESS_BOARD_COL_CNT};

    OnDiskSupervisor *sup = new OnDiskSupervisor(0.00f, 1.00f, board_name_buf, label_name_buf, x_shape, 0);

    return sup;
}

void train_chess(const char *pgn_name)
{
    OnDiskSupervisor *sup = get_chess_train_supervisor(pgn_name);

    Model *model = new Model(CostFunction::MSE, 0.001f);

    model->add_layer(new ConvolutionalLayer(sup->get_x_shape(), 256, 1, 1, InitializationFunction::Xavier));
    model->add_layer(new ConvolutionalLayer(model->get_output_shape(), 128, 3, 3, InitializationFunction::Xavier));
    model->add_layer(new ConvolutionalLayer(model->get_output_shape(), 128, 3, 3, InitializationFunction::Xavier));

    model->add_layer(new LinearLayer(model->get_output_shape(), 1024, InitializationFunction::Xavier));
    model->add_layer(new ActivationLayer(model->get_output_shape(), ActivationFunction::Tanh));

    model->add_layer(new LinearLayer(model->get_output_shape(), 1024, InitializationFunction::Xavier));
    model->add_layer(new ActivationLayer(model->get_output_shape(), ActivationFunction::Tanh));

    model->add_layer(new LinearLayer(model->get_output_shape(), 256, InitializationFunction::Xavier));
    model->add_layer(new ActivationLayer(model->get_output_shape(), ActivationFunction::Tanh));

    model->add_layer(new LinearLayer(model->get_output_shape(), 64, InitializationFunction::Xavier));
    model->add_layer(new ActivationLayer(model->get_output_shape(), ActivationFunction::Tanh));

    model->add_layer(new LinearLayer(model->get_output_shape(), Tensor::get_cnt(sup->get_y_shape()), InitializationFunction::Xavier));
    model->add_layer(new ActivationLayer(model->get_output_shape(), ActivationFunction::Tanh));

    model->train_and_test(sup, 64, 3, "C:\\Users\\danny\\Desktop\\chess-zero\\chess-zero-train.csv");

    model->save("C:\\Users\\danny\\Desktop\\chess-zero\\chess-zero.nn");

    delete model;

    delete sup;
}

void train_chess_existing(const char *pgn_name, const char *model_path)
{
    OnDiskSupervisor *sup = get_chess_train_supervisor(pgn_name);

    Model *model = new Model(model_path);
    model->set_learning_rate(0.0001f);

    model->train_and_test(sup, 64, 3, "C:\\Users\\danny\\Desktop\\chess-zero\\chess-zero-train.csv");

    model->save("C:\\Users\\danny\\Desktop\\chess-zero\\chess-zero-existing.nn");

    delete model;

    delete sup;
}

void test_chess(const char *pgn_name, const char *model_path)
{
    OnDiskSupervisor *sup = get_chess_test_supervisor(pgn_name);

    Model *model = new Model(model_path);

    Batch *test_batch = sup->create_test_batch();

    model->test(test_batch).print();

    delete test_batch;

    delete model;

    delete sup;
}

MoveSearchResult get_best_move(int *immut_board, bool white_mov_flg, bool print_flg, int depth, Model *model)
{
    int legal_moves[CHESS_MAX_LEGAL_MOVE_CNT];
    char mov[CHESS_MAX_MOVE_LEN];

    int sim_board[CHESS_BOARD_LEN];

    float flt_one_hot_board[CHESS_ONE_HOT_ENCODED_BOARD_LEN];

    char minimax_mov[CHESS_MAX_MOVE_LEN];
    float best_minimax_eval;
    MinimaxResult minimax_res;
    if (white_mov_flg)
    {
        best_minimax_eval = -100.0f;
    }
    else
    {
        best_minimax_eval = 100.0f;
    }

    float best_model_eval;
    if (white_mov_flg)
    {
        best_model_eval = -100.0f;
    }
    else
    {
        best_model_eval = 100.0f;
    }

    float depth_0_minimax_eval = eval_board(immut_board, model);

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

                        // Minimax evaluation:
                        {
                            minimax_res = get_minimax(sim_board, white_mov_flg, !white_mov_flg, depth, 1, depth_0_minimax_eval, best_minimax_eval);
                        }

                        if (print_flg)
                        {
                            printf("%s\t%f\t%d\n", mov, minimax_res.eval, minimax_res.prune_flg);
                        }

                        if (minimax_res.eval > best_minimax_eval)
                        {
                            best_minimax_eval = minimax_res.eval;
                            memcpy(minimax_mov, mov, CHESS_MAX_MOVE_LEN);

                            // Model:
                            {
                                float flt_one_hot_board[CHESS_ONE_HOT_ENCODED_BOARD_LEN];

                                one_hot_encode_board(immut_board, flt_one_hot_board);
                                std::vector<int> x_shape{CHESS_ONE_HOT_ENCODE_COMBINATION_CNT, CHESS_BOARD_ROW_CNT, CHESS_BOARD_COL_CNT};
                                Tensor *x = new Tensor(Device::Cpu, x_shape);
                                x->set_arr(flt_one_hot_board);
                                Tensor *pred = model->predict(x);
                                best_model_eval = pred->get_val(0);
                                delete pred;
                                delete x;
                            }
                        }
                        else if (minimax_res.eval == best_minimax_eval)
                        {
                            // Model:
                            float model_eval;
                            {
                                float flt_one_hot_board[CHESS_ONE_HOT_ENCODED_BOARD_LEN];

                                one_hot_encode_board(immut_board, flt_one_hot_board);
                                std::vector<int> x_shape{CHESS_ONE_HOT_ENCODE_COMBINATION_CNT, CHESS_BOARD_ROW_CNT, CHESS_BOARD_COL_CNT};
                                Tensor *x = new Tensor(Device::Cpu, x_shape);
                                x->set_arr(flt_one_hot_board);
                                Tensor *pred = model->predict(x);
                                model_eval = pred->get_val(0);
                                delete pred;
                                delete x;
                            }

                            if (model_eval > best_model_eval)
                            {
                                best_model_eval = model_eval;
                                memcpy(minimax_mov, mov, CHESS_MAX_MOVE_LEN);
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
                        simulate_board_change_w_srcdst_idx(immut_board, piece_idx, legal_moves[mov_idx], sim_board);

                        // Move string:
                        {
                            memset(mov, 0, CHESS_MAX_MOVE_LEN);
                            translate_srcdst_idx_to_mov(immut_board, piece_idx, legal_moves[mov_idx], mov);
                        }

                        // Minimax evaluation:
                        {
                            minimax_res = get_minimax(sim_board, white_mov_flg, !white_mov_flg, depth, 1, depth_0_minimax_eval, best_minimax_eval);
                        }

                        if (print_flg)
                        {
                            printf("%s\t%f\t%d\n", mov, minimax_res.eval, minimax_res.prune_flg);
                        }

                        if (minimax_res.eval < best_minimax_eval)
                        {
                            best_minimax_eval = minimax_res.eval;
                            memcpy(minimax_mov, mov, CHESS_MAX_MOVE_LEN);

                            // Model:
                            {
                                float flt_one_hot_board[CHESS_ONE_HOT_ENCODED_BOARD_LEN];

                                one_hot_encode_board(immut_board, flt_one_hot_board);
                                std::vector<int> x_shape{CHESS_ONE_HOT_ENCODE_COMBINATION_CNT, CHESS_BOARD_ROW_CNT, CHESS_BOARD_COL_CNT};
                                Tensor *x = new Tensor(Device::Cpu, x_shape);
                                x->set_arr(flt_one_hot_board);
                                Tensor *pred = model->predict(x);
                                best_model_eval = pred->get_val(0);
                                delete pred;
                                delete x;
                            }
                        }
                        else if (minimax_res.eval == best_minimax_eval)
                        {
                            // Model:
                            float model_eval;
                            {
                                float flt_one_hot_board[CHESS_ONE_HOT_ENCODED_BOARD_LEN];

                                one_hot_encode_board(immut_board, flt_one_hot_board);
                                std::vector<int> x_shape{CHESS_ONE_HOT_ENCODE_COMBINATION_CNT, CHESS_BOARD_ROW_CNT, CHESS_BOARD_COL_CNT};
                                Tensor *x = new Tensor(Device::Cpu, x_shape);
                                x->set_arr(flt_one_hot_board);
                                Tensor *pred = model->predict(x);
                                model_eval = pred->get_val(0);
                                delete pred;
                                delete x;
                            }

                            if (model_eval < best_model_eval)
                            {
                                best_model_eval = model_eval;
                                memcpy(minimax_mov, mov, CHESS_MAX_MOVE_LEN);
                            }
                        }
                    }
                }
            }
        }
    }

    if (print_flg)
    {
        printf("-------+---------------+---------------\n");
    }

    MoveSearchResult mov_res;
    memcpy(mov_res.mov, minimax_mov, CHESS_MAX_MOVE_LEN);
    mov_res.minimax_eval = best_minimax_eval;
    return mov_res;
}

void play_chess(const char *model_path, bool white_flg, int depth, bool print_flg)
{
    int *board = init_board();
    int cpy_board[CHESS_BOARD_LEN];
    char mov[CHESS_MAX_MOVE_LEN];

    Model *model = new Model(model_path);

    bool white_mov_flg = true;

    int opening_idx = 0;

    std::vector<Opening> openings = get_pgn_openings("train");

    if (white_flg)
    {
        std::sort(openings.begin(), openings.end(), opening_white_wins_sort_func);
    }
    else
    {
        std::sort(openings.begin(), openings.end(), opening_black_wins_sort_func);
    }

    // Opening:
    for (int mov_idx = 0; mov_idx < CHESS_START_MOVE_IDX; mov_idx++)
    {

        if (white_flg)
        {
            // White move:
            print_board(board);
            memcpy(board, &openings[opening_idx].all_board_states[mov_idx * CHESS_BOARD_LEN],
                   sizeof(int) * (CHESS_BOARD_LEN));
            system("cls");
            white_mov_flg = !white_mov_flg;
            mov_idx++;

            // Black move now:
            print_flipped_board(board);
            memset(mov, 0, CHESS_MAX_MOVE_LEN);
            printf("BLACK: ");
            std::cin >> mov;
            system("cls");
            change_board_w_mov(board, mov, white_mov_flg);
            white_mov_flg = !white_mov_flg;

            // Check opening match:
            bool opening_match_flg = true;
            while (memcmp(board, &openings[opening_idx].all_board_states[mov_idx * CHESS_BOARD_LEN],
                          sizeof(int) * CHESS_BOARD_LEN) != 0)
            {
                opening_idx++;

                if (opening_idx >= openings.size())
                {
                    opening_match_flg = false;
                    break;
                }
            }

            if (!opening_match_flg)
            {
                break;
            }
        }
        else
        {
            // White move:
            print_board(board);
            memset(mov, 0, CHESS_MAX_MOVE_LEN);
            printf("WHITE: ");
            std::cin >> mov;
            system("cls");
            change_board_w_mov(board, mov, white_mov_flg);
            white_mov_flg = !white_mov_flg;

            // Check opening match:
            bool opening_match_flg = true;
            while (memcmp(board, &openings[opening_idx].all_board_states[mov_idx * CHESS_BOARD_LEN],
                          sizeof(int) * CHESS_BOARD_LEN) != 0)
            {
                opening_idx++;

                if (opening_idx >= openings.size())
                {
                    opening_match_flg = false;
                    break;
                }
            }

            if (!opening_match_flg)
            {
                // Make black move from model since no opening match:
                {
                    print_flipped_board(board);

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

                        printf("move\tminimax\t\tpruned\n");
                        printf("-------+---------------+------------\n");

                        MoveSearchResult mov_res;

                        copy_board(board, cpy_board);
                        mov_res = get_best_move(cpy_board, white_mov_flg, print_flg, depth, model);
                        printf("%s\t%f\t-\n", mov_res.mov, mov_res.minimax_eval);

                        printf("-------+---------------+------------\n");

                        // Now accept user input:
                        memset(mov, 0, CHESS_MAX_MOVE_LEN);
                        printf("BLACK (a OR <custom>): ");
                        std::cin >> mov;
                        system("cls");

                        // Allow user to confirm they want to make a recommended move.
                        if (strcmp(mov, "a") == 0)
                        {
                            strcpy(mov, mov_res.mov);
                        }

                        change_board_w_mov(board, mov, white_mov_flg);
                        white_mov_flg = !white_mov_flg;
                    }
                }

                break;
            }

            mov_idx++;

            // Black move now:
            print_flipped_board(board);
            memcpy(board, &openings[opening_idx].all_board_states[mov_idx * CHESS_BOARD_LEN],
                   sizeof(int) * (CHESS_BOARD_LEN));
            system("cls");
            white_mov_flg = !white_mov_flg;
        }
    }

    system("cls");

    // Middle/end:
    while (true)
    {

        print_board(board);

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

            if (white_flg)
            {
                printf("move\tminimax\t\tpruned\n");
                printf("-------+---------------+------------\n");

                MoveSearchResult mov_res;

                copy_board(board, cpy_board);
                mov_res = get_best_move(cpy_board, white_mov_flg, print_flg, depth, model);
                printf("%s\t%f\t-\n", mov_res.mov, mov_res.minimax_eval);

                printf("-------+---------------+------------\n");

                // Now accept user input:
                memset(mov, 0, CHESS_MAX_MOVE_LEN);
                printf("WHITE (a OR <custom>): ");
                std::cin >> mov;
                system("cls");

                // Allow user to confirm they want to make a recommended move.
                if (strcmp(mov, "a") == 0)
                {
                    strcpy(mov, mov_res.mov);
                }
            }
            else
            {
                // Now accept user input:
                memset(mov, 0, CHESS_MAX_MOVE_LEN);
                printf("WHITE: ");
                std::cin >> mov;
                system("cls");
            }

            change_board_w_mov(board, mov, white_mov_flg);
            white_mov_flg = !white_mov_flg;
        }

        print_flipped_board(board);

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

            if (!white_flg)
            {
                printf("move\tminimax\t\tpruned\n");
                printf("-------+---------------+------------\n");

                MoveSearchResult mov_res;

                copy_board(board, cpy_board);
                mov_res = get_best_move(cpy_board, white_mov_flg, print_flg, depth, model);
                printf("%s\t%f\t-\n", mov_res.mov, mov_res.minimax_eval);

                printf("-------+---------------+------------\n");

                // Now accept user input:
                memset(mov, 0, CHESS_MAX_MOVE_LEN);
                printf("BLACK (a OR <custom>): ");
                std::cin >> mov;
                system("cls");

                // Allow user to confirm they want to make a recommended move.
                if (strcmp(mov, "a") == 0)
                {
                    strcpy(mov, mov_res.mov);
                }
            }
            else
            {
                // Now accept user input:
                memset(mov, 0, CHESS_MAX_MOVE_LEN);
                printf("BLACK: ");
                std::cin >> mov;
                system("cls");
            }

            change_board_w_mov(board, mov, white_mov_flg);
            white_mov_flg = !white_mov_flg;
        }
    }

    free(board);
}

int main(int argc, char **argv)
{
    srand(time(NULL));

    //dump_pgn("test");

    //train_chess("train");

    //train_chess_existing("train");

    //test_chess("test", "C:\\Users\\danny\\Desktop\\chess-zero\\chess-zero.nn");

    play_chess("C:\\Users\\danny\\Desktop\\chess-zero\\chess-zero.nn", false, 4, false);

    return 0;
}