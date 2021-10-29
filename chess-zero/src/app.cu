
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
    int worst_case;
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
    sprintf(file_name_buf, "c:\\users\\d0g0825\\desktop\\temp\\chess-zero\\white-%s.bs", pgn_name);
    FILE *white_boards_file = fopen(file_name_buf, "wb");

    memset(file_name_buf, 0, 256);
    sprintf(file_name_buf, "c:\\users\\d0g0825\\desktop\\temp\\chess-zero\\black-%s.bs", pgn_name);
    FILE *black_boards_file = fopen(file_name_buf, "wb");

    bool white_mov_flg;

    int *board = init_board();
    int influence_board[CHESS_BOARD_LEN];

    // Skip openings!
    int start_mov_idx = 10;

    printf("Total Games: %d\n", pgn->cnt);

    for (int game_idx = 0; game_idx < pgn->cnt; game_idx++)
    {
        PGNMoveList *pl = pgn->games[game_idx];

        {
            white_mov_flg = true;

            FILE *boards_file = nullptr;

            for (int mov_idx = 0; mov_idx < pl->cnt; mov_idx++)
            {
                if (white_mov_flg)
                {
                    boards_file = white_boards_file;
                }
                else
                {
                    boards_file = black_boards_file;
                }

                if (mov_idx >= start_mov_idx)
                {
                    // Write pre board state.
                    fwrite(board, sizeof(int) * CHESS_BOARD_LEN, 1, boards_file);
                    // Save pre move influence board.
                    get_influence_board(board, influence_board);

                    // Make move.
                    change_board_w_mov(board, pl->arr[mov_idx], white_mov_flg);

                    // Write post move board state.
                    fwrite(board, sizeof(int) * CHESS_BOARD_LEN, 1, boards_file);

                    // Write pre move influence board state.
                    fwrite(influence_board, sizeof(int) * CHESS_BOARD_LEN, 1, boards_file);
                    // Get and save post move influence board.
                    get_influence_board(board, influence_board);
                    fwrite(influence_board, sizeof(int) * CHESS_BOARD_LEN, 1, boards_file);
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

    fclose(white_boards_file);
    fclose(black_boards_file);

    PGNImport_free(pgn);

    system("cls");
}

Supervisor *get_chess_supervisor(const char *pgn_name)
{
    char file_name_buf[256];

    FILE *boards_file;
    long long boards_file_size;

    memset(file_name_buf, 0, 256);
    sprintf(file_name_buf, "c:\\users\\d0g0825\\desktop\\temp\\chess-zero\\%s.bs", pgn_name);
    boards_file = fopen(file_name_buf, "rb");
    boards_file_size = get_chess_file_size(file_name_buf);

    FILE *labels_file;
    long long labels_file_size;

    memset(file_name_buf, 0, 256);
    sprintf(file_name_buf, "c:\\users\\d0g0825\\desktop\\temp\\chess-zero\\%s.ls", pgn_name);
    labels_file = fopen(file_name_buf, "rb");
    labels_file_size = get_chess_file_size(file_name_buf);

    int file_col_cnt = CHESS_BOARD_LEN * 4;
    int file_row_cnt = (boards_file_size / (sizeof(int) * file_col_cnt));

    int *board_buf = (int *)malloc(sizeof(int) * (file_row_cnt * file_col_cnt));
    fread(board_buf, sizeof(int), (file_row_cnt * file_col_cnt), boards_file);

    float *board_flt_buf = (float *)malloc(sizeof(float) * (file_row_cnt * file_col_cnt));
    for (int i = 0; i < (file_row_cnt * file_col_cnt); i++)
    {
        board_flt_buf[i] = ((float)board_buf[i] / (((int)ChessPiece::WhiteKing) * 1.0f));
    }

    free(board_buf);

    float *label_flt_buf = (float *)malloc(sizeof(float) * (file_row_cnt));
    fread(label_flt_buf, sizeof(int), (file_row_cnt), labels_file);

    Supervisor *sup = new Supervisor(file_row_cnt, file_col_cnt, 0, board_flt_buf,
                                     label_flt_buf, 0.8f, 0.2f, Device::Cpu);

    free(board_flt_buf);
    free(label_flt_buf);

    fclose(boards_file);
    fclose(labels_file);

    sup->shuffle();

    return sup;
}

void train_chess(const char *pgn_name)
{
    Supervisor *sup = get_chess_supervisor(pgn_name);

    // TRAIN NEW ================================= =================================

    // Model *model = new Model(CostFunction::MSE, 0.1f);

    // std::vector<int> n_shape{2, CHESS_BOARD_ROW_CNT, CHESS_BOARD_COL_CNT};

    // model->add_layer(new ConvolutionalLayer(n_shape, 64, 5, 5, InitializationFunction::Xavier));
    // model->add_layer(new ActivationLayer(model->get_output_shape(), ActivationFunction::ReLU));

    // model->add_layer(new LinearLayer(model->get_output_shape(), 128, InitializationFunction::Xavier));
    // model->add_layer(new ActivationLayer(model->get_output_shape(), ActivationFunction::ReLU));

    // model->add_layer(new LinearLayer(model->get_output_shape(), 64, InitializationFunction::Xavier));
    // model->add_layer(new ActivationLayer(model->get_output_shape(), ActivationFunction::ReLU));

    // model->add_layer(new LinearLayer(model->get_output_shape(), Tensor::get_cnt(sup->get_y_shape()), InitializationFunction::Xavier));
    // model->add_layer(new ActivationLayer(model->get_output_shape(), ActivationFunction::Tanh));

    // model->train_and_test(sup, 256, 10, "C:\\Users\\d0g0825\\Desktop\\temp\\chess-zero\\chess.csv");

    // model->save("C:\\Users\\d0g0825\\Desktop\\temp\\chess-zero\\chess.nn");

    // TRAIN EXISTING ================================= =================================

    Model *model = new Model("C:\\Users\\d0g0825\\Desktop\\temp\\chess-zero\\chess.nn");

    model->train_and_test(sup, 256, 10, "C:\\Users\\d0g0825\\Desktop\\temp\\chess-zero\\chess.csv");

    model->save("C:\\Users\\d0g0825\\Desktop\\temp\\chess-zero\\chess.nn");

    //  ================================= =================================

    delete model;

    delete sup;
}

void train_chess_custom(const char *pgn_name)
{
    char file_name_buf[256];

    FILE *white_boards_file;
    long long white_boards_file_size;

    memset(file_name_buf, 0, 256);
    sprintf(file_name_buf, "c:\\users\\d0g0825\\desktop\\temp\\chess-zero\\white-%s.bs", pgn_name);
    white_boards_file = fopen(file_name_buf, "rb");
    white_boards_file_size = get_chess_file_size(file_name_buf);

    int white_file_col_cnt = CHESS_BOARD_LEN * 4;
    int white_file_row_cnt = (white_boards_file_size / (sizeof(int) * white_file_col_cnt));

    FILE *black_boards_file;
    long long black_boards_file_size;

    memset(file_name_buf, 0, 256);
    sprintf(file_name_buf, "c:\\users\\d0g0825\\desktop\\temp\\chess-zero\\black-%s.bs", pgn_name);
    black_boards_file = fopen(file_name_buf, "rb");
    black_boards_file_size = get_chess_file_size(file_name_buf);

    int black_file_col_cnt = CHESS_BOARD_LEN * 4;
    int black_file_row_cnt = (black_boards_file_size / (sizeof(int) * black_file_col_cnt));

    int board[CHESS_BOARD_LEN];
    int post_mov_board[CHESS_BOARD_LEN];
    int influence_board[CHESS_BOARD_LEN];
    int post_mov_influence_board[CHESS_BOARD_LEN];
    int board_buf[CHESS_BOARD_LEN * 4];

    float flt_board[CHESS_BOARD_LEN];
    float flt_post_mov_board[CHESS_BOARD_LEN];
    float flt_influence_board[CHESS_BOARD_LEN];
    float flt_post_mov_influence_board[CHESS_BOARD_LEN];
    float flt_board_buf[CHESS_BOARD_LEN * 4];

    FILE *csv_file_ptr = fopen("C:\\Users\\d0g0825\\Desktop\\temp\\chess-zero\\chess.csv", "w");
    CSVUtils::write_csv_header(csv_file_ptr);

    Model *model = new Model(CostFunction::CrossEntropy, 0.1f);

    std::vector<int> n_shape{2, CHESS_BOARD_ROW_CNT * 2, CHESS_BOARD_COL_CNT};

    model->add_layer(new ConvolutionalLayer(n_shape, 64, 5, 5, InitializationFunction::Xavier));
    model->add_layer(new ActivationLayer(model->get_output_shape(), ActivationFunction::ReLU));

    model->add_layer(new LinearLayer(model->get_output_shape(), 128, InitializationFunction::Xavier));
    model->add_layer(new ActivationLayer(model->get_output_shape(), ActivationFunction::ReLU));

    model->add_layer(new LinearLayer(model->get_output_shape(), 64, InitializationFunction::Xavier));
    model->add_layer(new ActivationLayer(model->get_output_shape(), ActivationFunction::ReLU));

    model->add_layer(new LinearLayer(model->get_output_shape(), 2, InitializationFunction::Xavier));
    model->add_layer(new ActivationLayer(model->get_output_shape(), ActivationFunction::Sigmoid));

    int record_idx = 0;

    bool white_turn_flg = true;

    long long int iteration = 0;

    while (true)
    {
        FILE *boards_file = nullptr;
        long long boards_file_size = 0;
        int file_col_cnt = 0;
        int file_row_cnt = 0;

        if (white_turn_flg)
        {
            boards_file = white_boards_file;
            boards_file_size = white_boards_file_size;
            file_col_cnt = white_file_col_cnt;
            file_row_cnt = white_file_row_cnt;
        }
        else
        {
            boards_file = black_boards_file;
            boards_file_size = black_boards_file_size;
            file_col_cnt = black_file_col_cnt;
            file_row_cnt = black_file_row_cnt;
        }

        record_idx = rand() % file_row_cnt;

        fseek(boards_file, record_idx * (CHESS_BOARD_LEN * 4), SEEK_SET);

        fread(board_buf, sizeof(int), CHESS_BOARD_LEN * 4, boards_file);

        memcpy(board, board_buf, sizeof(int) * CHESS_BOARD_LEN);
        memcpy(influence_board, &board_buf[CHESS_BOARD_LEN], sizeof(int) * CHESS_BOARD_LEN);
        memcpy(post_mov_board, &board_buf[CHESS_BOARD_LEN * 2], sizeof(int) * CHESS_BOARD_LEN);
        memcpy(post_mov_influence_board, &board_buf[CHESS_BOARD_LEN * 3], sizeof(int) * CHESS_BOARD_LEN);

        board_to_float(board, flt_board, true);
        board_to_float(influence_board, flt_influence_board, true);
        board_to_float(post_mov_board, flt_post_mov_board, true);
        board_to_float(post_mov_influence_board, flt_post_mov_influence_board, true);

        memcpy(flt_board_buf, flt_board, sizeof(float) * CHESS_BOARD_LEN);
        memcpy(&flt_board_buf[CHESS_BOARD_LEN], flt_influence_board, sizeof(float) * CHESS_BOARD_LEN);
        memcpy(&flt_board_buf[CHESS_BOARD_LEN * 2], flt_post_mov_board, sizeof(float) * CHESS_BOARD_LEN);
        memcpy(&flt_board_buf[CHESS_BOARD_LEN * 3], flt_post_mov_influence_board, sizeof(float) * CHESS_BOARD_LEN);

        Tensor *x = new Tensor(Device::Cpu, 2, CHESS_BOARD_ROW_CNT * 2, CHESS_BOARD_COL_CNT);
        x->set_arr(flt_board_buf);

        Tensor *y = new Tensor(Device::Cpu, 2);
        y->set_val(0, 1.0f);
        y->set_val(1, 0.0f);

        Batch *batch = new Batch(true, 3);
        batch->add(new Record(x, y));

        for (int i = 0; i < 2; i++)
        {
            SrcDst_Idx sdi = get_random_move(board, white_turn_flg, post_mov_board);
            simulate_board_change_w_srcdst_idx(board, sdi.src_idx, sdi.dst_idx, post_mov_board);

            get_influence_board(post_mov_board, post_mov_influence_board);

            board_to_float(post_mov_board, flt_post_mov_board, true);
            board_to_float(post_mov_influence_board, flt_post_mov_influence_board, true);

            memcpy(&flt_board_buf[CHESS_BOARD_LEN * 2], flt_post_mov_board, sizeof(float) * CHESS_BOARD_LEN);
            memcpy(&flt_board_buf[CHESS_BOARD_LEN * 3], flt_post_mov_influence_board, sizeof(float) * CHESS_BOARD_LEN);

            Tensor *_x = new Tensor(Device::Cpu, 2, CHESS_BOARD_ROW_CNT * 2, CHESS_BOARD_COL_CNT);
            _x->set_arr(flt_board_buf);

            Tensor *_y = new Tensor(Device::Cpu, 2);
            _y->set_val(0, 0.0f);
            _y->set_val(1, 1.0f);

            batch->add(new Record(_x, _y));
        }

        Report rpt = model->train(batch);

        int epoch = iteration / (white_file_row_cnt + black_file_row_cnt);

        CSVUtils::write_to_csv(csv_file_ptr, epoch, iteration, rpt);

        delete batch;

        white_turn_flg = !white_turn_flg;

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

        iteration++;
    }

    model->save("C:\\Users\\d0g0825\\Desktop\\temp\\chess-zero\\chess.nn");

    delete model;

    fclose(white_boards_file);
    fclose(black_boards_file);

    fclose(csv_file_ptr);
}

MoveSearchResult get_best_move(int *immut_board, bool white_mov_flg, bool print_flg, int depth, Model *model)
{
    int legal_moves[CHESS_MAX_LEGAL_MOVE_CNT];
    char mov[CHESS_MAX_MOVE_LEN];

    int sim_board[CHESS_BOARD_LEN];
    int influence_board[CHESS_BOARD_LEN];
    int post_mov_influence_board[CHESS_BOARD_LEN];

    float flt_board[CHESS_BOARD_LEN];
    float flt_post_mov_board[CHESS_BOARD_LEN];
    float flt_influence_board[CHESS_BOARD_LEN];
    float flt_post_mov_influence_board[CHESS_BOARD_LEN];
    float flt_board_buf[CHESS_BOARD_LEN * 4];

    float best_eval = -FLT_MAX;

    int best_worst_case;
    if (white_mov_flg)
    {
        best_worst_case = -FLT_MAX;
    }
    else
    {
        best_worst_case = FLT_MAX;
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
                        simulate_board_change_w_srcdst_idx(immut_board, piece_idx, legal_moves[mov_idx], sim_board);
                        get_influence_board(immut_board, influence_board);
                        get_influence_board(sim_board, post_mov_influence_board);

                        board_to_float(immut_board, flt_board, true);
                        board_to_float(influence_board, flt_influence_board, true);
                        board_to_float(sim_board, flt_post_mov_board, true);
                        board_to_float(post_mov_influence_board, flt_post_mov_influence_board, true);

                        memcpy(flt_board_buf, flt_board, sizeof(float) * CHESS_BOARD_LEN);
                        memcpy(&flt_board_buf[CHESS_BOARD_LEN], flt_influence_board, sizeof(float) * CHESS_BOARD_LEN);
                        memcpy(&flt_board_buf[CHESS_BOARD_LEN * 2], flt_post_mov_board, sizeof(float) * CHESS_BOARD_LEN);
                        memcpy(&flt_board_buf[CHESS_BOARD_LEN * 3], flt_post_mov_influence_board, sizeof(float) * CHESS_BOARD_LEN);

                        Tensor *x = new Tensor(Device::Cpu, 2, CHESS_BOARD_ROW_CNT * 2, CHESS_BOARD_COL_CNT);
                        x->set_arr(flt_board_buf);

                        Tensor *pred = model->predict(x);

                        memset(mov, 0, CHESS_MAX_MOVE_LEN);
                        translate_srcdst_idx_to_mov(immut_board, piece_idx, legal_moves[mov_idx], mov);

                        if (print_flg)
                        {
                            printf("MOVE: %s (%f\t%f)\n", mov, pred->get_val(0), pred->get_val(1));
                            pred->print();
                        }

                        delete pred;

                        // if (best_worst_case < eval)
                        // {
                        //     best_worst_case = eval;
                        //     best_eval = eval;
                        //     memcpy(best_mov, mov, CHESS_MAX_MOVE_LEN);
                        // }

                        // if (best_worst_case == eval)
                        // {
                        //     if (best_eval < eval)
                        //     {
                        //         best_eval = eval;
                        //         memcpy(best_mov, mov, CHESS_MAX_MOVE_LEN);
                        //     }
                        // }
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
                        get_influence_board(immut_board, influence_board);
                        get_influence_board(sim_board, post_mov_influence_board);

                        board_to_float(immut_board, flt_board, true);
                        board_to_float(influence_board, flt_influence_board, true);
                        board_to_float(sim_board, flt_post_mov_board, true);
                        board_to_float(post_mov_influence_board, flt_post_mov_influence_board, true);

                        memcpy(flt_board_buf, flt_board, sizeof(float) * CHESS_BOARD_LEN);
                        memcpy(&flt_board_buf[CHESS_BOARD_LEN], flt_influence_board, sizeof(float) * CHESS_BOARD_LEN);
                        memcpy(&flt_board_buf[CHESS_BOARD_LEN * 2], flt_post_mov_board, sizeof(float) * CHESS_BOARD_LEN);
                        memcpy(&flt_board_buf[CHESS_BOARD_LEN * 3], flt_post_mov_influence_board, sizeof(float) * CHESS_BOARD_LEN);

                        Tensor *x = new Tensor(Device::Cpu, 2, CHESS_BOARD_ROW_CNT * 2, CHESS_BOARD_COL_CNT);
                        x->set_arr(flt_board_buf);

                        Tensor *pred = model->predict(x);

                        memset(mov, 0, CHESS_MAX_MOVE_LEN);
                        translate_srcdst_idx_to_mov(immut_board, piece_idx, legal_moves[mov_idx], mov);

                        if (print_flg)
                        {
                            printf("MOVE: %s (%f\t%f)\n", mov, pred->get_val(0), pred->get_val(1));
                        }

                        delete pred;

                        // if (best_worst_case > eval)
                        // {
                        //     best_worst_case = eval;
                        //     best_eval = eval;
                        //     memcpy(best_mov, mov, CHESS_MAX_MOVE_LEN);
                        // }

                        // if (best_worst_case == eval)
                        // {
                        //     if (best_eval < eval)
                        //     {
                        //         best_eval = eval;
                        //         memcpy(best_mov, mov, CHESS_MAX_MOVE_LEN);
                        //     }
                        // }
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

void play_chess(const char *model_path, bool white_flg, int depth)
{
    Model *model = new Model(model_path);

    int *board = init_board();
    int inf_board[CHESS_BOARD_LEN];
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
                mov_res = get_best_move(cpy_board, white_mov_flg, true, depth, model);
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
                mov_res = get_best_move(cpy_board, white_mov_flg, false, depth, model);
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
    srand(time(NULL));

    //dump_pgn("TEST");

    //train_chess_custom("TEST");

    //train_chess("TEST");

    play_chess("C:\\Users\\d0g0825\\Desktop\\temp\\chess-zero\\chess.nn", true, 1);

    return 0;
}