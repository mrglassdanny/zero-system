
#include <iostream>
#include <windows.h>

#include <zero_system/nn/model.cuh>

#include "chess.cuh"
#include "pgn.cuh"

using namespace zero::core;
using namespace zero::nn;

#define CHESS_START_MOVE_IDX 10
#define CHESS_SELF_PLAY_MAX_MOV_CNT 270

struct MoveSearchResult
{
    char mov[CHESS_MAX_MOVE_LEN];
    float eval;
};

struct Opening
{
    int all_board_states[CHESS_BOARD_LEN * CHESS_START_MOVE_IDX];
    int white_win_cnt;
    int black_win_cnt;
    int tie_cnt;
};

class Game
{
public:
    std::vector<Tensor *> board_states;
    std::vector<float> evals;
    float lbl;

    Game()
    {
        this->lbl = 0.0f;
    }

    ~Game()
    {
        for (int i = 0; i < this->board_states.size(); i++)
        {
            delete this->board_states[i];
        }
    }
};

void dump_pgn(const char *pgn_name)
{
    char file_name_buf[256];
    memset(file_name_buf, 0, 256);
    sprintf(file_name_buf, "data\\%s.pgn", pgn_name);
    PGNImport *pgn = PGNImport_init(file_name_buf);

    memset(file_name_buf, 0, 256);
    sprintf(file_name_buf, "temp\\%s.bs", pgn_name);
    FILE *boards_file = fopen(file_name_buf, "wb");

    memset(file_name_buf, 0, 256);
    sprintf(file_name_buf, "temp\\%s.bl", pgn_name);
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
    sprintf(board_name_buf, "temp\\%s.bs", pgn_name);

    memset(label_name_buf, 0, 256);
    sprintf(label_name_buf, "temp\\%s.bl", pgn_name);

    std::vector<int> x_shape{CHESS_ONE_HOT_ENCODE_COMBINATION_CNT, CHESS_BOARD_ROW_CNT, CHESS_BOARD_COL_CNT};

    OnDiskSupervisor *sup = new OnDiskSupervisor(1.00f, 0.00f, board_name_buf, label_name_buf, x_shape, 0);

    return sup;
}

OnDiskSupervisor *get_chess_test_supervisor(const char *pgn_name)
{
    char board_name_buf[256];
    char label_name_buf[256];

    memset(board_name_buf, 0, 256);
    sprintf(board_name_buf, "temp\\%s.bs", pgn_name);

    memset(label_name_buf, 0, 256);
    sprintf(label_name_buf, "temp\\%s.bl", pgn_name);

    std::vector<int> x_shape{CHESS_ONE_HOT_ENCODE_COMBINATION_CNT, CHESS_BOARD_ROW_CNT, CHESS_BOARD_COL_CNT};

    OnDiskSupervisor *sup = new OnDiskSupervisor(0.00f, 1.00f, board_name_buf, label_name_buf, x_shape, 0);

    return sup;
}

Model *init_chess_model()
{
    Model *model = new Model(CostFunction::MSE, 0.01f);

    std::vector<int> x_shape{CHESS_ONE_HOT_ENCODE_COMBINATION_CNT, CHESS_BOARD_ROW_CNT, CHESS_BOARD_COL_CNT};
    int y_shape = 1;

    model->add_layer(new ConvolutionalLayer(x_shape, 1, 1, 1, InitializationFunction::Xavier));
    model->add_layer(new ActivationLayer(model->get_output_shape(), ActivationFunction::Tanh));

    // model->add_layer(new ConvolutionalLayer(model->get_output_shape(), 1, 8, 8, InitializationFunction::Xavier));
    // model->add_layer(new ActivationLayer(model->get_output_shape(), ActivationFunction::Tanh));

    model->add_layer(new LinearLayer(model->get_output_shape(), 512, InitializationFunction::Xavier));
    model->add_layer(new ActivationLayer(model->get_output_shape(), ActivationFunction::Tanh));

    model->add_layer(new LinearLayer(model->get_output_shape(), 128, InitializationFunction::Xavier));
    model->add_layer(new ActivationLayer(model->get_output_shape(), ActivationFunction::Tanh));

    model->add_layer(new LinearLayer(model->get_output_shape(), 16, InitializationFunction::Xavier));
    model->add_layer(new ActivationLayer(model->get_output_shape(), ActivationFunction::Tanh));

    model->add_layer(new LinearLayer(model->get_output_shape(), y_shape, InitializationFunction::Xavier));
    model->add_layer(new ActivationLayer(model->get_output_shape(), ActivationFunction::Tanh));

    return model;
}

MoveSearchResult get_best_move(int *immut_board, bool white_mov_flg, Model *model)
{
    int legal_moves[CHESS_MAX_LEGAL_MOVE_CNT];
    char mov[CHESS_MAX_MOVE_LEN];

    int sim_board[CHESS_BOARD_LEN];

    float flt_one_hot_board[CHESS_ONE_HOT_ENCODED_BOARD_LEN];

    char best_model_mov[CHESS_MAX_MOVE_LEN];
    float best_model_eval;
    if (white_mov_flg)
    {
        best_model_eval = -100.0f;
    }
    else
    {
        best_model_eval = 100.0f;
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

                        // Move string:
                        {
                            memset(mov, 0, CHESS_MAX_MOVE_LEN);
                            translate_srcdst_idx_to_mov(immut_board, piece_idx, legal_moves[mov_idx], mov);
                        }

                        // Model evaluation:
                        float model_eval;
                        {
                            one_hot_encode_board(immut_board, flt_one_hot_board);
                            std::vector<int> x_shape{CHESS_ONE_HOT_ENCODE_COMBINATION_CNT, CHESS_BOARD_ROW_CNT, CHESS_BOARD_COL_CNT};
                            Tensor *x = new Tensor(Device::Cpu, x_shape);
                            x->set_arr(flt_one_hot_board);
                            Tensor *pred = model->predict(x);
                            model_eval = pred->get_val(0);
                            delete pred;
                            delete x;
                        }

                        // Compare:
                        if (model_eval > best_model_eval)
                        {
                            best_model_eval = model_eval;
                            memcpy(best_model_mov, mov, CHESS_MAX_MOVE_LEN);
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
                        float model_eval;
                        {
                            one_hot_encode_board(immut_board, flt_one_hot_board);
                            std::vector<int> x_shape{CHESS_ONE_HOT_ENCODE_COMBINATION_CNT, CHESS_BOARD_ROW_CNT, CHESS_BOARD_COL_CNT};
                            Tensor *x = new Tensor(Device::Cpu, x_shape);
                            x->set_arr(flt_one_hot_board);
                            Tensor *pred = model->predict(x);
                            model_eval = pred->get_val(0);
                            delete pred;
                            delete x;
                        }

                        // Compare:
                        if (model_eval < best_model_eval)
                        {
                            best_model_eval = model_eval;
                            memcpy(best_model_mov, mov, CHESS_MAX_MOVE_LEN);
                        }
                    }
                }
            }
        }
    }

    MoveSearchResult mov_res;
    memcpy(mov_res.mov, best_model_mov, CHESS_MAX_MOVE_LEN);
    mov_res.eval = best_model_eval;
    return mov_res;
}

Game *play_chess(Model *model)
{
    Game *game = new Game();

    int *board = init_board();
    int rot_board[CHESS_BOARD_LEN];
    float flt_one_hot_board[CHESS_ONE_HOT_ENCODED_BOARD_LEN];

    std::vector<int> x_shape{CHESS_ONE_HOT_ENCODE_COMBINATION_CNT, CHESS_BOARD_ROW_CNT, CHESS_BOARD_COL_CNT};

    bool white_mov_flg = true;

    int mov_cnt = 0;

    while (true)
    {
        // White move:
        {
            if (is_in_checkmate(board, white_mov_flg))
            {
                print_board(board);
                game->lbl = -1.0f;
                break;
            }

            if (is_in_stalemate(board, white_mov_flg))
            {
                print_board(board);
                game->lbl = 0.0f;
                break;
            }

            MoveSearchResult mov_res = get_best_move(board, white_mov_flg, model);

            change_board_w_mov(board, mov_res.mov, white_mov_flg);
            white_mov_flg = !white_mov_flg;

            one_hot_encode_board(board, flt_one_hot_board);
            Tensor *x = new Tensor(Device::Cpu, x_shape);
            x->set_arr(flt_one_hot_board);
            game->board_states.push_back(x);

            // Rotate board:

            // rotate_board(board, rot_board, 90);
            // one_hot_encode_board(rot_board, flt_one_hot_board);
            // Tensor *x2 = new Tensor(Device::Cpu, x_shape);
            // x2->set_arr(flt_one_hot_board);
            // game->board_states.push_back(x2);

            // rotate_board(board, rot_board, 180);
            // one_hot_encode_board(rot_board, flt_one_hot_board);
            // Tensor *x3 = new Tensor(Device::Cpu, x_shape);
            // x3->set_arr(flt_one_hot_board);
            // game->board_states.push_back(x3);

            // rotate_board(board, rot_board, 270);
            // one_hot_encode_board(rot_board, flt_one_hot_board);
            // Tensor *x4 = new Tensor(Device::Cpu, x_shape);
            // x4->set_arr(flt_one_hot_board);
            // game->board_states.push_back(x4);

            mov_cnt++;
        }

        // Black move:
        {
            if (is_in_checkmate(board, white_mov_flg))
            {
                print_board(board);
                game->lbl = 1.0f;
                break;
            }

            if (is_in_stalemate(board, white_mov_flg))
            {
                print_board(board);
                game->lbl = 0.0f;
                break;
            }

            MoveSearchResult mov_res = get_best_move(board, white_mov_flg, model);

            change_board_w_mov(board, mov_res.mov, white_mov_flg);
            white_mov_flg = !white_mov_flg;

            one_hot_encode_board(board, flt_one_hot_board);
            Tensor *x = new Tensor(Device::Cpu, x_shape);
            x->set_arr(flt_one_hot_board);
            game->board_states.push_back(x);

            // Rotate board:

            // rotate_board(board, rot_board, 90);
            // one_hot_encode_board(rot_board, flt_one_hot_board);
            // Tensor *x2 = new Tensor(Device::Cpu, x_shape);
            // x2->set_arr(flt_one_hot_board);
            // game->board_states.push_back(x2);

            // rotate_board(board, rot_board, 180);
            // one_hot_encode_board(rot_board, flt_one_hot_board);
            // Tensor *x3 = new Tensor(Device::Cpu, x_shape);
            // x3->set_arr(flt_one_hot_board);
            // game->board_states.push_back(x3);

            // rotate_board(board, rot_board, 270);
            // one_hot_encode_board(rot_board, flt_one_hot_board);
            // Tensor *x4 = new Tensor(Device::Cpu, x_shape);
            // x4->set_arr(flt_one_hot_board);
            // game->board_states.push_back(x4);

            mov_cnt++;
        }

        // Check if tie:
        {
            // 3 move repetition:
            int board_cnt = game->board_states.size();
            if (board_cnt > 6)
            {
                Tensor *x_1 = game->board_states[board_cnt - 1];
                Tensor *x_2 = game->board_states[board_cnt - 3];
                Tensor *x_3 = game->board_states[board_cnt - 5];

                if (x_1->equals(x_2) && x_1->equals(x_3))
                {
                    printf("3 move repetition!\n");
                    game->lbl = 0.0f;
                    break;
                }
            }

            // Exceeds move count:
            if (mov_cnt >= CHESS_SELF_PLAY_MAX_MOV_CNT)
            {
                printf("Move count exceeded!\n");
                game->lbl = 0.0f;
                break;
            }
        }
    }

    free(board);

    return game;
}

void train_chess(Model *model, Game *game)
{
    Tensor *y = new Tensor(Device::Cuda, 1);

    y->set_val(0, game->lbl);

    float cost = 0.0f;

    for (int i = 0; i < game->board_states.size(); i++)
    {
        Tensor *pred = model->forward(game->board_states[i], true);
        cost += model->cost(pred, y);
        model->backward(pred, y);
        delete pred;
    }

    printf("Cost: %f\n", cost / game->board_states.size());

    model->step(game->board_states.size());

    delete y;
}

int main(int argc, char **argv)
{
    srand(time(NULL));

    Model *model = init_chess_model();

    int white_win_cnt = 0;
    int black_win_cnt = 0;
    int tie_cnt = 0;

    int game_cnt = 0;
    while (true)
    {
        printf("Playing...\n");
        Game *game = play_chess(model);

        system("cls");

        printf("Training...\n");
        train_chess(model, game);

        if (game->lbl == 1.0f)
        {
            white_win_cnt++;
        }
        else if (game->lbl == -1.0f)
        {
            black_win_cnt++;
        }
        else
        {
            tie_cnt++;
        }

        delete game;

        if (_kbhit())
        {
            if (_getch() == 'q')
            {
                printf("Quitting...\n");
                break;
            }
        }

        printf("Games played: %d (%d - %d - %d)\n", ++game_cnt, white_win_cnt, black_win_cnt, tie_cnt);
    }

    model->save("temp\\chess-zero.nn");

    delete model;

    return 0;
}