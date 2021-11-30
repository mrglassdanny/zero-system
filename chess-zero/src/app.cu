
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

Model *init_chess_model()
{
    Model *model = new Model(CostFunction::MSE, 0.001f);

    std::vector<int> x_shape{CHESS_ONE_HOT_ENCODE_COMBINATION_CNT, CHESS_BOARD_ROW_CNT, CHESS_BOARD_COL_CNT};
    int y_shape = 1;

    model->add_layer(new ConvolutionalLayer(x_shape, 32, 1, 1, InitializationFunction::Xavier));
    model->add_layer(new ActivationLayer(model->get_output_shape(), ActivationFunction::Tanh));

    model->add_layer(new ConvolutionalLayer(model->get_output_shape(), 128, 8, 8, InitializationFunction::Xavier));
    model->add_layer(new ActivationLayer(model->get_output_shape(), ActivationFunction::Tanh));

    model->add_layer(new LinearLayer(model->get_output_shape(), 512, InitializationFunction::Xavier));
    model->add_layer(new ActivationLayer(model->get_output_shape(), ActivationFunction::Tanh));

    model->add_layer(new LinearLayer(model->get_output_shape(), 64, InitializationFunction::Xavier));
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
    float flt_one_hot_board[CHESS_ONE_HOT_ENCODED_BOARD_LEN];

    std::vector<int> x_shape{CHESS_ONE_HOT_ENCODE_COMBINATION_CNT, CHESS_BOARD_ROW_CNT, CHESS_BOARD_COL_CNT};

    bool white_mov_flg = true;

    int mov_cnt = 0;

    while (true)
    {
        printf("Playing...\n");
        // White move:
        {
            if (is_in_checkmate(board, white_mov_flg))
            {
                game->lbl = -1.0f;
                break;
            }

            if (is_in_stalemate(board, white_mov_flg))
            {
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

            // TODO: reflect/rotate board

            mov_cnt++;
        }

        // Black move:
        {
            if (is_in_checkmate(board, white_mov_flg))
            {
                game->lbl = 1.0f;
                break;
            }

            if (is_in_stalemate(board, white_mov_flg))
            {
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

            // TODO: reflect/rotate board

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
                    game->lbl = 0.0f;
                    break;
                }
            }

            // Exceeds turn count:
            if (mov_cnt >= 300)
            {
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

    printf("COST: %f\n", cost / game->board_states.size());

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
        system("cls");

        Game *game = play_chess(model);
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

        printf("%d (%d - %d - %d)\n", ++game_cnt, white_win_cnt, black_win_cnt, tie_cnt);
    }

    model->save("temp\\chess-zero.nn");

    delete model;

    return 0;
}