
#include <iostream>
#include <windows.h>

#include "chess.cuh"
#include "pgn.cuh"
#include "chess_model.cuh"

struct MoveSearchResult
{
    char mov[CHESS_MAX_MOVE_LEN];
    float minimax_eval;
};

void dump_pgn(const char *pgn_name)
{
    char file_name_buf[256];
    memset(file_name_buf, 0, 256);
    sprintf(file_name_buf, "c:\\users\\d0g0825\\ml-data\\chess-zero\\%s.pgn", pgn_name);
    PGNImport *pgn = PGNImport_init(file_name_buf);

    memset(file_name_buf, 0, 256);
    sprintf(file_name_buf, "c:\\users\\d0g0825\\desktop\\temp\\chess-zero\\%s.pbs", pgn_name);
    FILE *piece_boards_file = fopen(file_name_buf, "wb");

    memset(file_name_buf, 0, 256);
    sprintf(file_name_buf, "c:\\users\\d0g0825\\desktop\\temp\\chess-zero\\%s.pbl", pgn_name);
    FILE *piece_labels_file = fopen(file_name_buf, "wb");

    memset(file_name_buf, 0, 256);
    sprintf(file_name_buf, "c:\\users\\d0g0825\\desktop\\temp\\chess-zero\\%s.mbs", pgn_name);
    FILE *move_boards_file = fopen(file_name_buf, "wb");

    memset(file_name_buf, 0, 256);
    sprintf(file_name_buf, "c:\\users\\d0g0825\\desktop\\temp\\chess-zero\\%s.mbl", pgn_name);
    FILE *move_labels_file = fopen(file_name_buf, "wb");

    bool white_mov_flg;

    int *board = init_board();

    float flt_one_hot_board_piece[CHESS_ONE_HOT_ENCODED_BOARD_LEN];
    float flt_one_hot_board_move[CHESS_ONE_HOT_ENCODED_BOARD_LEN + CHESS_BOARD_LEN];

    float piece_lbl;
    float move_lbl;

    // Skip openings!
    int start_mov_idx = 0;

    printf("Total Games: %d\n", pgn->cnt);

    for (int game_idx = 0; game_idx < pgn->cnt; game_idx++)
    {
        PGNMoveList *pl = pgn->games[game_idx];

        {
            white_mov_flg = true;

            for (int mov_idx = 0; mov_idx < pl->cnt; mov_idx++)
            {
                if (mov_idx >= start_mov_idx)
                {
                    // Make move.
                    ChessMove chess_move = change_board_w_mov(board, pl->arr[mov_idx], white_mov_flg);

                    one_hot_encode_board(board, flt_one_hot_board_piece);
                    fwrite(flt_one_hot_board_piece, sizeof(float) * CHESS_ONE_HOT_ENCODED_BOARD_LEN, 1, piece_boards_file);

                    piece_lbl = (float)chess_move.src_idx;
                    fwrite(&piece_lbl, sizeof(float), 1, piece_labels_file);

                    memcpy(flt_one_hot_board_move, flt_one_hot_board_piece, sizeof(float) * CHESS_ONE_HOT_ENCODED_BOARD_LEN);
                    Tensor *p = Tensor::one_hot_encode(Device::Cpu, 1, CHESS_BOARD_LEN, &piece_lbl);
                    memcpy(&flt_one_hot_board_move[CHESS_ONE_HOT_ENCODED_BOARD_LEN], p->get_arr(), sizeof(float) * CHESS_BOARD_LEN);
                    fwrite(flt_one_hot_board_move, sizeof(float) * (CHESS_ONE_HOT_ENCODED_BOARD_LEN + CHESS_BOARD_LEN), 1, move_boards_file);
                    delete p;

                    move_lbl = (float)chess_move.dst_idx;
                    fwrite(&move_lbl, sizeof(float), 1, move_labels_file);
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

    fclose(piece_boards_file);
    fclose(piece_labels_file);
    fclose(move_boards_file);
    fclose(move_labels_file);

    PGNImport_free(pgn);

    system("cls");
}

OnDiskSupervisor *get_chess_piece_supervisor(const char *pgn_name)
{
    char board_name_buf[256];
    char label_name_buf[256];

    memset(board_name_buf, 0, 256);
    sprintf(board_name_buf, "c:\\users\\d0g0825\\desktop\\temp\\chess-zero\\%s.pbs", pgn_name);

    memset(label_name_buf, 0, 256);
    sprintf(label_name_buf, "c:\\users\\d0g0825\\desktop\\temp\\chess-zero\\%s.pbl", pgn_name);

    std::vector<int> x_shape{CHESS_ONE_HOT_ENCODE_COMBINATION_CNT, CHESS_BOARD_ROW_CNT, CHESS_BOARD_COL_CNT};

    OnDiskSupervisor *sup = new OnDiskSupervisor(0.90f, 0.10f, board_name_buf, label_name_buf, x_shape, CHESS_BOARD_LEN);

    return sup;
}

OnDiskSupervisor *get_chess_move_supervisor(const char *pgn_name)
{
    char board_name_buf[256];
    char label_name_buf[256];

    memset(board_name_buf, 0, 256);
    sprintf(board_name_buf, "c:\\users\\d0g0825\\desktop\\temp\\chess-zero\\%s.mbs", pgn_name);

    memset(label_name_buf, 0, 256);
    sprintf(label_name_buf, "c:\\users\\d0g0825\\desktop\\temp\\chess-zero\\%s.mbl", pgn_name);

    std::vector<int> x_shape{CHESS_ONE_HOT_ENCODE_COMBINATION_CNT + 1, CHESS_BOARD_ROW_CNT, CHESS_BOARD_COL_CNT};

    OnDiskSupervisor *sup = new OnDiskSupervisor(0.90f, 0.10f, board_name_buf, label_name_buf, x_shape, CHESS_BOARD_LEN);

    return sup;
}

void train_chess_piece(const char *pgn_name)
{
    OnDiskSupervisor *sup = get_chess_piece_supervisor(pgn_name);

    ChessModel *model = new ChessModel(CostFunction::CrossEntropy, 0.1f);

    model->add_layer(new ConvolutionalLayer(sup->get_x_shape(), 128, 3, 3, InitializationFunction::Xavier));
    model->add_layer(new ActivationLayer(model->get_output_shape(), ActivationFunction::ReLU));

    model->add_layer(new ConvolutionalLayer(model->get_output_shape(), 128, 3, 3, InitializationFunction::Xavier));
    model->add_layer(new ActivationLayer(model->get_output_shape(), ActivationFunction::ReLU));

    model->add_layer(new LinearLayer(model->get_output_shape(), 512, InitializationFunction::Xavier));
    model->add_layer(new ActivationLayer(model->get_output_shape(), ActivationFunction::ReLU));

    model->add_layer(new LinearLayer(model->get_output_shape(), Tensor::get_cnt(sup->get_y_shape()), InitializationFunction::Xavier));

    model->train_and_test(sup, 50, 10, "C:\\Users\\d0g0825\\Desktop\\temp\\chess-zero\\chess-piece.csv");

    model->save("C:\\Users\\d0g0825\\Desktop\\temp\\chess-zero\\chess-piece.nn");

    delete model;

    delete sup;
}

void train_chess_move(const char *pgn_name)
{
    OnDiskSupervisor *sup = get_chess_move_supervisor(pgn_name);

    ChessModel *model = new ChessModel(CostFunction::CrossEntropy, 0.1f);

    model->add_layer(new ConvolutionalLayer(sup->get_x_shape(), 128, 3, 3, InitializationFunction::Xavier));
    model->add_layer(new ActivationLayer(model->get_output_shape(), ActivationFunction::ReLU));

    model->add_layer(new ConvolutionalLayer(model->get_output_shape(), 128, 3, 3, InitializationFunction::Xavier));
    model->add_layer(new ActivationLayer(model->get_output_shape(), ActivationFunction::ReLU));

    model->add_layer(new LinearLayer(model->get_output_shape(), 512, InitializationFunction::Xavier));
    model->add_layer(new ActivationLayer(model->get_output_shape(), ActivationFunction::ReLU));

    model->add_layer(new LinearLayer(model->get_output_shape(), Tensor::get_cnt(sup->get_y_shape()), InitializationFunction::Xavier));

    model->train_and_test(sup, 50, 10, "C:\\Users\\d0g0825\\Desktop\\temp\\chess-zero\\chess-move.csv");

    model->save("C:\\Users\\d0g0825\\Desktop\\temp\\chess-zero\\chess-move.nn");

    delete model;

    delete sup;
}

MoveSearchResult get_best_move(int *immut_board, bool white_mov_flg, bool print_flg, int depth, ChessModel *piece_model, ChessModel *move_model)
{
    int legal_moves[CHESS_MAX_LEGAL_MOVE_CNT];
    char mov[CHESS_MAX_MOVE_LEN];

    int sim_board[CHESS_BOARD_LEN];

    float flt_one_hot_board_piece[CHESS_ONE_HOT_ENCODED_BOARD_LEN];
    float flt_one_hot_board_move[CHESS_ONE_HOT_ENCODED_BOARD_LEN + CHESS_BOARD_LEN];

    float best_minimax_eval;
    if (white_mov_flg)
    {
        best_minimax_eval = -FLT_MAX;
    }
    else
    {
        best_minimax_eval = FLT_MAX;
    }

    char best_mov[CHESS_MAX_MOVE_LEN];

    one_hot_encode_board(immut_board, flt_one_hot_board_piece);

    Tensor *x = new Tensor(Device::Cpu, CHESS_ONE_HOT_ENCODE_COMBINATION_CNT, CHESS_BOARD_ROW_CNT, CHESS_BOARD_COL_CNT);
    x->set_arr(flt_one_hot_board_piece);

    // Predict piece:
    piece_model->set_piece_legality_mask(x, white_mov_flg);
    Tensor *piece_pred = piece_model->predict_piece(x);

    TensorTuple piece_tup = piece_pred->get_max();
    float piece_eval = piece_tup.val;
    printf("Piece %c%d (%c)", get_col_fr_idx(piece_tup.idx), get_row_fr_idx(piece_tup.idx), get_char_fr_piece((ChessPiece)immut_board[piece_tup.idx]));
    delete piece_pred;

    {
        float piece_idx = (float)piece_tup.idx;
        memcpy(flt_one_hot_board_move, flt_one_hot_board_piece, sizeof(float) * CHESS_ONE_HOT_ENCODED_BOARD_LEN);
        Tensor *p = Tensor::one_hot_encode(Device::Cpu, 1, CHESS_BOARD_LEN, &piece_idx);
        memcpy(&flt_one_hot_board_move[CHESS_ONE_HOT_ENCODED_BOARD_LEN], p->get_arr(), sizeof(float) * CHESS_BOARD_LEN);
        delete p;

        delete x;
        x = new Tensor(Device::Cpu, CHESS_ONE_HOT_ENCODE_COMBINATION_CNT + 1, CHESS_BOARD_ROW_CNT, CHESS_BOARD_COL_CNT);
        x->set_arr(flt_one_hot_board_move);
    }

    // Predict move:
    move_model->set_move_legality_mask(x, piece_tup.idx);
    Tensor *move_pred = move_model->predict_move(x);

    TensorTuple move_tup = move_pred->get_max();
    float move_eval = move_tup.val;
    printf(" to %c%d\n", get_col_fr_idx(move_tup.idx), get_row_fr_idx(move_tup.idx));
    delete move_pred;

    if (print_flg)
    {
        printf("move\tminimax\t\tpruned\n");
        printf("-------+---------------+------------\n");
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

                        memset(mov, 0, CHESS_MAX_MOVE_LEN);
                        translate_srcdst_idx_to_mov(immut_board, piece_idx, legal_moves[mov_idx], mov);

                        MinimaxEvaluation minimax_eval = get_minimax_eval(sim_board, white_mov_flg, !white_mov_flg, depth, 1, best_minimax_eval);

                        if (print_flg)
                        {
                            printf("%s\t%f\t%d\n", mov, minimax_eval.eval, minimax_eval.prune_flg);
                        }

                        if (minimax_eval.eval > best_minimax_eval)
                        {
                            best_minimax_eval = minimax_eval.eval;
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

                        memset(mov, 0, CHESS_MAX_MOVE_LEN);
                        translate_srcdst_idx_to_mov(immut_board, piece_idx, legal_moves[mov_idx], mov);

                        MinimaxEvaluation minimax_eval = get_minimax_eval(sim_board, white_mov_flg, !white_mov_flg, depth, 1, best_minimax_eval);

                        if (print_flg)
                        {
                            printf("%s\t%f\t%d\n", mov, minimax_eval.eval, minimax_eval.prune_flg);
                        }

                        if (minimax_eval.eval < best_minimax_eval)
                        {
                            best_minimax_eval = minimax_eval.eval;
                            memcpy(best_mov, mov, CHESS_MAX_MOVE_LEN);
                        }
                    }
                }
            }
        }
    }

    if (print_flg)
    {
        printf("-------+---------------+------------\n");
    }

    MoveSearchResult mov_res;
    memcpy(mov_res.mov, best_mov, CHESS_MAX_MOVE_LEN);
    mov_res.minimax_eval = best_minimax_eval;
    return mov_res;
}

void play_chess(const char *piece_model_path, const char *move_model_path, bool white_flg, int depth, bool print_flg)
{
    ChessModel *piece_model = new ChessModel(piece_model_path);
    ChessModel *move_model = new ChessModel(move_model_path);

    int *board = init_board();
    int cpy_board[CHESS_BOARD_LEN];
    char mov[CHESS_MAX_MOVE_LEN];

    bool white_mov_flg = true;

    // // Go ahead and make opening moves since we do not train the model on openings.
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

            copy_board(board, cpy_board);
            mov_res = get_best_move(cpy_board, white_mov_flg, print_flg, depth, piece_model, move_model);
            printf("MoveSearchResult: %s\t(%f)\n", mov_res.mov, mov_res.minimax_eval);

            // Now accept user input.
            memset(mov, 0, CHESS_MAX_MOVE_LEN);
            printf("ENTER MOVE (WHITE): ");

            std::cin >> mov;
            system("cls");

            // Allow user to confirm they want to make recommended move.
            if (strlen(mov) <= 1)
            {
                strcpy(mov, mov_res.mov);
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

            copy_board(board, cpy_board);
            mov_res = get_best_move(cpy_board, white_mov_flg, print_flg, depth, piece_model, move_model);
            printf("MoveSearchResult: %s\t(%f)\n", mov_res.mov, mov_res.minimax_eval);

            // Now accept user input.
            memset(mov, 0, CHESS_MAX_MOVE_LEN);
            printf("ENTER MOVE (BLACK): ");
            std::cin >> mov;
            system("cls");

            // Allow user to confirm they want to make recommended move.
            if (strlen(mov) <= 1)
            {
                strcpy(mov, mov_res.mov);
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

    //dump_pgn("ALL");

    train_chess_piece("ALL");
    train_chess_move("ALL");

    //play_chess("C:\\Users\\d0g0825\\Desktop\\temp\\chess-zero\\chess-piece.nn", "C:\\Users\\d0g0825\\Desktop\\temp\\chess-zero\\chess-move.nn", true, 3, true);

    return 0;
}