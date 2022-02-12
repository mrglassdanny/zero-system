
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

ConvNet *init_model()
{
    ConvNet *conv = new ConvNet(CostFunction::MSE, 0.01f);

    std::vector<int> x_shape{CHESS_ONE_HOT_ENCODE_COMBINATION_CNT, CHESS_BOARD_ROW_CNT, CHESS_BOARD_COL_CNT};
    int y_shape = 1;

    conv->convolutional(x_shape, 1, 1, 1);
    conv->activation(ActivationFunction::Tanh);

    conv->linear(1024);
    conv->activation(ActivationFunction::Tanh);

    conv->linear(1024);
    conv->activation(ActivationFunction::Tanh);

    conv->linear(128);
    conv->activation(ActivationFunction::Tanh);

    conv->linear(y_shape);
    conv->activation(ActivationFunction::Tanh);

    return conv;
}

Model *init_model(const char *model_path)
{
    Model *model = new Model(model_path);
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

Game *self_play(Model *model)
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
                game->lbl = -1.0f;
                printf("Black won!\n");
                break;
            }

            if (is_in_stalemate(board, white_mov_flg))
            {
                game->lbl = 0.0f;
                printf("Stalemate!\n");
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
            {
                rotate_board(board, rot_board, 90);
                one_hot_encode_board(rot_board, flt_one_hot_board);
                Tensor *x2 = new Tensor(Device::Cpu, x_shape);
                x2->set_arr(flt_one_hot_board);
                game->board_states.push_back(x2);

                rotate_board(board, rot_board, 180);
                one_hot_encode_board(rot_board, flt_one_hot_board);
                Tensor *x3 = new Tensor(Device::Cpu, x_shape);
                x3->set_arr(flt_one_hot_board);
                game->board_states.push_back(x3);

                rotate_board(board, rot_board, 270);
                one_hot_encode_board(rot_board, flt_one_hot_board);
                Tensor *x4 = new Tensor(Device::Cpu, x_shape);
                x4->set_arr(flt_one_hot_board);
                game->board_states.push_back(x4);
            }

            mov_cnt++;
        }

        // Black move:
        {
            if (is_in_checkmate(board, white_mov_flg))
            {
                game->lbl = 1.0f;
                printf("White won!\n");
                break;
            }

            if (is_in_stalemate(board, white_mov_flg))
            {
                game->lbl = 0.0f;
                printf("Stalemate!\n");
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
            {
                rotate_board(board, rot_board, 90);
                one_hot_encode_board(rot_board, flt_one_hot_board);
                Tensor *x2 = new Tensor(Device::Cpu, x_shape);
                x2->set_arr(flt_one_hot_board);
                game->board_states.push_back(x2);

                rotate_board(board, rot_board, 180);
                one_hot_encode_board(rot_board, flt_one_hot_board);
                Tensor *x3 = new Tensor(Device::Cpu, x_shape);
                x3->set_arr(flt_one_hot_board);
                game->board_states.push_back(x3);

                rotate_board(board, rot_board, 270);
                one_hot_encode_board(rot_board, flt_one_hot_board);
                Tensor *x4 = new Tensor(Device::Cpu, x_shape);
                x4->set_arr(flt_one_hot_board);
                game->board_states.push_back(x4);
            }

            mov_cnt++;
        }

        // Check if tie:
        {
            // 3 move repetition:
            int board_cnt = game->board_states.size();
            if (board_cnt > 24)
            {
                Tensor *x_1 = game->board_states[board_cnt - 1];
                Tensor *x_2 = game->board_states[board_cnt - 9];
                Tensor *x_3 = game->board_states[board_cnt - 17];

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

void self_train(Model *model, Game *game)
{
    Tensor *y = new Tensor(Device::Cuda, 1);

    y->set_val(0, game->lbl);

    float cost = 0.0f;

    for (int i = 0; i < game->board_states.size(); i++)
    {
        Tensor *pred = model->forward(game->board_states[i], true);
        cost += model->cost(pred, y);
        delete model->backward(pred, y);
        delete pred;
    }

    printf("Cost: %f\n", cost / game->board_states.size());

    model->step(game->board_states.size());

    delete y;
}

void bootstrap_learn(Model *model)
{
    PGNImport *pgn = PGNImport_init("data\\bootstrap.pgn");

    int *board = init_board();
    int rot_board[CHESS_BOARD_LEN];
    float flt_one_hot_board[CHESS_ONE_HOT_ENCODED_BOARD_LEN];

    std::vector<int> x_shape{CHESS_ONE_HOT_ENCODE_COMBINATION_CNT, CHESS_BOARD_ROW_CNT, CHESS_BOARD_COL_CNT};

    bool white_mov_flg = true;

    int mov_cnt = 0;

    Tensor *x = new Tensor(Device::Cuda, x_shape);
    Tensor *y = new Tensor(Device::Cuda, 1);

    float cost = 0.0f;

    for (int game_idx = 0; game_idx < pgn->cnt; game_idx++)
    {
        PGNMoveList *pl = pgn->games[game_idx];

        {
            if (pl->white_won_flg)
            {
                y->set_val(0, 1.0f);
            }
            else if (pl->black_won_flg)
            {
                y->set_val(0, -1.0f);
            }
            else
            {
                y->set_val(0, 0.0f);
            }

            white_mov_flg = true;

            for (int mov_idx = 0; mov_idx < pl->cnt; mov_idx++)
            {
                change_board_w_mov(board, pl->arr[mov_idx], white_mov_flg);

                one_hot_encode_board(board, flt_one_hot_board);
                x->set_arr(flt_one_hot_board);

                Tensor *pred = model->forward(x, true);
                cost += model->cost(pred, y);
                delete model->backward(pred, y);
                delete pred;

                // Rotate board:
                {

                    rotate_board(board, rot_board, 90);
                    one_hot_encode_board(rot_board, flt_one_hot_board);
                    x->set_arr(flt_one_hot_board);
                    Tensor *pred2 = model->forward(x, true);
                    cost += model->cost(pred2, y);
                    delete model->backward(pred2, y);
                    delete pred2;

                    rotate_board(board, rot_board, 180);
                    one_hot_encode_board(rot_board, flt_one_hot_board);
                    x->set_arr(flt_one_hot_board);
                    Tensor *pred3 = model->forward(x, true);
                    cost += model->cost(pred3, y);
                    delete model->backward(pred3, y);
                    delete pred3;

                    rotate_board(board, rot_board, 270);
                    one_hot_encode_board(rot_board, flt_one_hot_board);
                    x->set_arr(flt_one_hot_board);
                    Tensor *pred4 = model->forward(x, true);
                    cost += model->cost(pred4, y);
                    delete model->backward(pred4, y);
                    delete pred4;
                }

                white_mov_flg = !white_mov_flg;
            }

            cost /= (pl->cnt * 4);
            model->step(pl->cnt * 4);

            reset_board(board);

            system("cls");
            printf("Game: %d / %d (%f)\n", game_idx, pgn->cnt, cost);
            cost = 0.0f;
        }
    }

    delete x;
    delete y;

    free(board);

    PGNImport_free(pgn);

    system("cls");
}

void self_learn(Model *model)
{
    int game_cnt = 0;
    int white_win_cnt = 0;
    int black_win_cnt = 0;
    int tie_cnt = 0;

    while (true)
    {
        printf("Playing...\n");
        Game *game = self_play(model);

        system("cls");

        printf("Training...\n");
        self_train(model, game);

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
}

void play_model(Model *model, bool model_white_flg)
{
    int *board = init_board();
    char mov[CHESS_MAX_MOVE_LEN];

    bool white_mov_flg = true;

    while (true)
    {

        // White move:
        {
            if (is_in_checkmate(board, white_mov_flg))
            {
                printf("Checkmate -- Black won!\n");
                print_board(board);
                break;
            }

            if (is_in_stalemate(board, white_mov_flg))
            {
                printf("Stalemate!\n");
                print_board(board);
                break;
            }

            if (is_in_check(board, true))
            {
                printf("Check!\n");
            }

            if (model_white_flg)
            {
                MoveSearchResult mov_res = get_best_move(board, white_mov_flg, model);
                change_board_w_mov(board, mov_res.mov, white_mov_flg);
                white_mov_flg = !white_mov_flg;
            }
            else
            {
                print_board(board);
                memset(mov, 0, CHESS_MAX_MOVE_LEN);
                printf("White Move: ");
                std::cin >> mov;
                system("cls");
                change_board_w_mov(board, mov, white_mov_flg);
                white_mov_flg = !white_mov_flg;
            }
        }

        system("cls");

        // Black move:
        {

            if (is_in_checkmate(board, white_mov_flg))
            {
                printf("Checkmate -- White won!\n");
                print_board(board);
                break;
            }

            if (is_in_stalemate(board, white_mov_flg))
            {
                printf("Stalemate!\n");
                print_board(board);
                break;
            }

            if (is_in_check(board, true))
            {
                printf("Check!\n");
            }

            if (!model_white_flg)
            {
                MoveSearchResult mov_res = get_best_move(board, white_mov_flg, model);
                change_board_w_mov(board, mov_res.mov, white_mov_flg);
                white_mov_flg = !white_mov_flg;
            }
            else
            {
                print_flipped_board(board);
                memset(mov, 0, CHESS_MAX_MOVE_LEN);
                printf("Black Move: ");
                std::cin >> mov;
                system("cls");
                change_board_w_mov(board, mov, white_mov_flg);
                white_mov_flg = !white_mov_flg;
            }
        }

        system("cls");
    }

    free(board);
}

int main(int argc, char **argv)
{
    srand(time(NULL));

    // Model *model = init_model();
    Model *model = init_model("temp\\bootstrapped-chess-zero.nn");

    // bootstrap_learn(model);

    // self_learn(model);

    play_model(model, false);

    // model->save("temp\\chess-zero.nn");

    delete model;

    return 0;
}