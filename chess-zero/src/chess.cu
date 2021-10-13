#include "chess.cuh"

int CHESS_START_BOARD[CHESS_BOARD_LEN] = {
    WhiteRook, WhiteKnight, WhiteBishop, WhiteQueen, WhiteKing, WhiteBishop, WhiteKnight, WhiteRook,
    WhitePawn, WhitePawn, WhitePawn, WhitePawn, WhitePawn, WhitePawn, WhitePawn, WhitePawn,
    Empty, Empty, Empty, Empty, Empty, Empty, Empty, Empty,
    Empty, Empty, Empty, Empty, Empty, Empty, Empty, Empty,
    Empty, Empty, Empty, Empty, Empty, Empty, Empty, Empty,
    Empty, Empty, Empty, Empty, Empty, Empty, Empty, Empty,
    BlackPawn, BlackPawn, BlackPawn, BlackPawn, BlackPawn, BlackPawn, BlackPawn, BlackPawn,
    BlackRook, BlackKnight, BlackBishop, BlackQueen, BlackKing, BlackBishop, BlackKnight, BlackRook};

int *init_board()
{
    int *board = (int *)malloc(sizeof(int) * (CHESS_BOARD_LEN));
    memcpy(board, CHESS_START_BOARD, sizeof(int) * (CHESS_BOARD_LEN));
    return board;
}

int *copy_board(int *src, int *dst)
{
    if (dst == NULL)
    {
        dst = (int *)malloc(sizeof(int) * (CHESS_BOARD_LEN));
    }

    memcpy(dst, src, sizeof(int) * CHESS_BOARD_LEN);
    return dst;
}

void reset_board(int *board)
{
    memcpy(board, CHESS_START_BOARD, sizeof(int) * (CHESS_BOARD_LEN));
}

int get_col_fr_adj_col(int adj_col)
{
    char col;

    switch (adj_col)
    {
    case 0:
        col = 'a';
        break;
    case 1:
        col = 'b';
        break;
    case 2:
        col = 'c';
        break;
    case 3:
        col = 'd';
        break;
    case 4:
        col = 'e';
        break;
    case 5:
        col = 'f';
        break;
    case 6:
        col = 'g';
        break;
    default:
        col = 'h';
        break;
    }

    return col;
}

int get_adj_col_fr_col(char col)
{
    int adj_col = 0;
    switch (col)
    {
    case 'a':
        adj_col = 0;
        break;
    case 'b':
        adj_col = 1;
        break;
    case 'c':
        adj_col = 2;
        break;
    case 'd':
        adj_col = 3;
        break;
    case 'e':
        adj_col = 4;
        break;
    case 'f':
        adj_col = 5;
        break;
    case 'g':
        adj_col = 6;
        break;
    default:
        adj_col = 7;
        break;
    }

    return adj_col;
}

int get_row_fr_char(char row)
{
    return (row - '0');
}

int get_adj_row_fr_row(int row)
{
    return row - 1;
}

int get_adj_col_fr_idx(int idx)
{
    return idx % CHESS_BOARD_COL_CNT;
}

int get_adj_row_fr_idx(int idx)
{
    return idx / CHESS_BOARD_ROW_CNT;
}

char get_col_fr_idx(int idx)
{
    int adj_col = get_adj_col_fr_idx(idx);
    switch (adj_col)
    {
    case 0:
        return 'a';
    case 1:
        return 'b';
    case 2:
        return 'c';
    case 3:
        return 'd';
    case 4:
        return 'e';
    case 5:
        return 'f';
    case 6:
        return 'g';
    default:
        return 'h';
    }
}

int get_row_fr_idx(int idx)
{
    return get_adj_row_fr_idx(idx) + 1;
}

int get_idx_fr_colrow(char col, int row)
{
    int adj_col = get_adj_col_fr_col(col);

    int adj_row = get_adj_row_fr_row(row);

    return (adj_row * CHESS_BOARD_COL_CNT) + adj_col;
}

int get_idx_fr_adj_colrow(int adj_col, int adj_row)
{
    return (adj_row * CHESS_BOARD_COL_CNT) + adj_col;
}

bool is_row_valid(int row)
{
    if (row >= 1 && row <= CHESS_BOARD_ROW_CNT)
    {
        return true;
    }
    else
    {
        return false;
    }
}

bool is_adj_colrow_valid(int adj_col, int adj_row)
{
    if (adj_col >= 0 && adj_col < CHESS_BOARD_COL_CNT &&
        adj_row >= 0 && adj_row < CHESS_BOARD_ROW_CNT)
    {
        return true;
    }
    else
    {
        return false;
    }
}

ChessPiece get_piece_fr_char(char piece_id, bool white_mov_flg)
{
    switch (piece_id)
    {
    case 'N':
        if (white_mov_flg)
        {
            return WhiteKnight;
        }
        else
        {
            return BlackKnight;
        }
    case 'B':
        if (white_mov_flg)
        {
            return WhiteBishop;
        }
        else
        {
            return BlackBishop;
        }
    case 'R':
        if (white_mov_flg)
        {
            return WhiteRook;
        }
        else
        {
            return BlackRook;
        }
    case 'Q':
        if (white_mov_flg)
        {
            return WhiteQueen;
        }
        else
        {
            return BlackQueen;
        }
    case 'K':
        if (white_mov_flg)
        {
            return WhiteKing;
        }
        else
        {
            return BlackKing;
        }
    default:
        // Pawn will be 'P' (optional).
        if (white_mov_flg)
        {
            return WhitePawn;
        }
        else
        {
            return BlackPawn;
        }
    }
}

char get_char_fr_piece(ChessPiece piece)
{
    switch (piece)
    {
    case WhiteKnight:
    case BlackKnight:
        return 'N';
    case WhiteBishop:
    case BlackBishop:
        return 'B';
    case WhiteRook:
    case BlackRook:
        return 'R';
    case WhiteQueen:
    case BlackQueen:
        return 'Q';
    case WhiteKing:
    case BlackKing:
        return 'K';
    default:
        // Pawn.
        return 'P';
    }
}

bool is_piece_white(ChessPiece piece)
{
    if (piece > 0)
    {
        return true;
    }
    else
    {
        return false;
    }
}

bool is_piece_black(ChessPiece piece)
{
    if (piece < 0)
    {
        return true;
    }
    else
    {
        return false;
    }
}

bool is_piece_same_color(ChessPiece a, ChessPiece b)
{
    if ((is_piece_white(a) && is_piece_white(b)) || (is_piece_black(a) && is_piece_black(b)))
    {
        return true;
    }
    else
    {
        return false;
    }
}

bool is_cell_under_attack(int *board, int cell_idx, bool white_pov_flg)
{
    bool under_attack_flg = false;
    int legal_moves[CHESS_MAX_LEGAL_MOVE_CNT];
    memset(legal_moves, CHESS_INVALID_VALUE, sizeof(int) * CHESS_MAX_LEGAL_MOVE_CNT);

    if (white_pov_flg)
    {
        for (int piece_idx = 0; piece_idx < CHESS_BOARD_LEN; piece_idx++)
        {
            if (is_piece_black((ChessPiece)board[piece_idx]))
            {
                get_legal_moves(board, piece_idx, legal_moves, true);

                for (int mov_idx = 0; mov_idx < CHESS_MAX_LEGAL_MOVE_CNT; mov_idx++)
                {
                    if (legal_moves[mov_idx] == CHESS_INVALID_VALUE)
                    {
                        break;
                    }

                    if (legal_moves[mov_idx] == cell_idx)
                    {
                        under_attack_flg = true;
                        break;
                    }
                }
            }

            if (under_attack_flg)
            {
                break;
            }
        }
    }
    else
    {
        for (int piece_idx = 0; piece_idx < CHESS_BOARD_LEN; piece_idx++)
        {
            if (is_piece_white((ChessPiece)board[piece_idx]))
            {
                get_legal_moves(board, piece_idx, legal_moves, true);

                for (int mov_idx = 0; mov_idx < CHESS_MAX_LEGAL_MOVE_CNT; mov_idx++)
                {
                    if (legal_moves[mov_idx] == CHESS_INVALID_VALUE)
                    {
                        break;
                    }

                    if (legal_moves[mov_idx] == cell_idx)
                    {
                        under_attack_flg = true;
                        break;
                    }
                }
            }

            if (under_attack_flg)
            {
                break;
            }
        }
    }

    return under_attack_flg;
}

bool is_in_check(int *board, bool white_mov_flg)
{
    bool in_check_flg = 0;
    int legal_moves[CHESS_MAX_LEGAL_MOVE_CNT];
    memset(legal_moves, CHESS_INVALID_VALUE, sizeof(int) * CHESS_MAX_LEGAL_MOVE_CNT);

    if (white_mov_flg)
    {
        for (int piece_idx = 0; piece_idx < CHESS_BOARD_LEN; piece_idx++)
        {
            if (is_piece_black((ChessPiece)board[piece_idx]))
            {
                get_legal_moves(board, piece_idx, legal_moves, false);

                for (int mov_idx = 0; mov_idx < CHESS_MAX_LEGAL_MOVE_CNT; mov_idx++)
                {
                    if (legal_moves[mov_idx] == CHESS_INVALID_VALUE)
                    {
                        break;
                    }

                    if ((ChessPiece)board[legal_moves[mov_idx]] == WhiteKing)
                    {
                        in_check_flg = true;
                        break;
                    }
                }
            }

            if (in_check_flg)
            {
                break;
            }
        }
    }
    else
    {
        for (int piece_idx = 0; piece_idx < CHESS_BOARD_LEN; piece_idx++)
        {
            if (is_piece_white((ChessPiece)board[piece_idx]))
            {
                get_legal_moves(board, piece_idx, legal_moves, false);

                for (int mov_idx = 0; mov_idx < CHESS_MAX_LEGAL_MOVE_CNT; mov_idx++)
                {
                    if (legal_moves[mov_idx] == CHESS_INVALID_VALUE)
                    {
                        break;
                    }

                    if ((ChessPiece)board[legal_moves[mov_idx]] == BlackKing)
                    {
                        in_check_flg = true;
                        break;
                    }
                }
            }

            if (in_check_flg)
            {
                break;
            }
        }
    }

    return in_check_flg;
}

bool is_in_checkmate(int *board, bool white_mov_flg)
{
    bool in_checkmate_flg;
    int legal_moves[CHESS_MAX_LEGAL_MOVE_CNT];
    memset(legal_moves, CHESS_INVALID_VALUE, sizeof(int) * CHESS_MAX_LEGAL_MOVE_CNT);

    if (is_in_check(board, white_mov_flg))
    {
        in_checkmate_flg = true;

        if (white_mov_flg)
        {
            for (int piece_idx = 0; piece_idx < CHESS_BOARD_LEN; piece_idx++)
            {
                if (is_piece_white((ChessPiece)board[piece_idx]))
                {
                    get_legal_moves(board, piece_idx, legal_moves, true);

                    if (legal_moves[0] != CHESS_INVALID_VALUE)
                    {
                        in_checkmate_flg = false;
                        break;
                    }
                }
            }
        }
        else
        {
            for (int piece_idx = 0; piece_idx < CHESS_BOARD_LEN; piece_idx++)
            {
                if (is_piece_black((ChessPiece)board[piece_idx]))
                {
                    get_legal_moves(board, piece_idx, legal_moves, true);

                    if (legal_moves[0] != CHESS_INVALID_VALUE)
                    {
                        in_checkmate_flg = false;
                        break;
                    }
                }
            }
        }
    }
    else
    {
        in_checkmate_flg = false;
    }

    return in_checkmate_flg;
}

void get_legal_moves(int *board, int piece_idx, int *out, bool test_in_check_flg)
{
    memset(out, CHESS_INVALID_VALUE, sizeof(int) * CHESS_MAX_LEGAL_MOVE_CNT);
    int mov_ctr = 0;

    ChessPiece piece = (ChessPiece)board[piece_idx];

    char col = get_col_fr_idx(piece_idx);
    int row = get_row_fr_idx(piece_idx);

    int adj_col = get_adj_col_fr_idx(piece_idx);
    int adj_row = get_adj_row_fr_idx(piece_idx);

    int white_mov_flg;
    if (is_piece_white(piece))
    {
        white_mov_flg = 1;
    }
    else
    {
        white_mov_flg = 0;
    }

    int test_idx;

    switch (piece)
    {
    case WhitePawn:
        // TODO: au passant
        {
            test_idx = get_idx_fr_colrow(col, row + 1);
            if (is_row_valid(row + 1) && board[test_idx] == Empty)
            {
                out[mov_ctr++] = test_idx;
            }

            test_idx = get_idx_fr_adj_colrow(adj_col - 1, adj_row + 1);
            if (is_adj_colrow_valid(adj_col - 1, adj_row + 1) && board[test_idx] != Empty && !is_piece_same_color(piece, (ChessPiece)board[test_idx]))
            {
                out[mov_ctr++] = test_idx;
            }

            test_idx = get_idx_fr_adj_colrow(adj_col + 1, adj_row + 1);
            if (is_adj_colrow_valid(adj_col + 1, adj_row + 1) && board[test_idx] != Empty && !is_piece_same_color(piece, (ChessPiece)board[test_idx]))
            {
                out[mov_ctr++] = test_idx;
            }

            if (row == 2)
            {
                // Dont need to check if row adjustments are valid since we know that starting row is 2.
                test_idx = get_idx_fr_colrow(col, row + 1);
                if (board[test_idx] == Empty)
                {
                    test_idx = get_idx_fr_colrow(col, row + 2);
                    if (board[test_idx] == Empty)
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }
            }
        }

        break;
    case BlackPawn:
        // TODO: au passant
        {
            test_idx = get_idx_fr_colrow(col, row - 1);
            if (is_row_valid(row - 1) && board[test_idx] == Empty)
            {
                out[mov_ctr++] = test_idx;
            }

            test_idx = get_idx_fr_adj_colrow(adj_col - 1, adj_row - 1);
            if (is_adj_colrow_valid(adj_col - 1, adj_row - 1) && board[test_idx] != Empty && !is_piece_same_color(piece, (ChessPiece)board[test_idx]))
            {
                out[mov_ctr++] = test_idx;
            }

            test_idx = get_idx_fr_adj_colrow(adj_col + 1, adj_row - 1);
            if (is_adj_colrow_valid(adj_col + 1, adj_row - 1) && board[test_idx] != Empty && !is_piece_same_color(piece, (ChessPiece)board[test_idx]))
            {
                out[mov_ctr++] = test_idx;
            }

            if (row == 7)
            {
                // Dont need to check if row adjustments are valid since we know that starting row is 7.
                test_idx = get_idx_fr_colrow(col, row - 1);
                if (board[test_idx] == Empty)
                {
                    test_idx = get_idx_fr_colrow(col, row - 2);
                    if (board[test_idx] == Empty)
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }
            }
        }

        break;
    case WhiteKnight:
    case BlackKnight:
    {

        if (is_adj_colrow_valid(adj_col + 1, adj_row + 2))
        {
            test_idx = get_idx_fr_adj_colrow(adj_col + 1, adj_row + 2);
            if (!is_piece_same_color(piece, (ChessPiece)board[test_idx]))
            {
                out[mov_ctr++] = test_idx;
            }
        }

        if (is_adj_colrow_valid(adj_col + 1, adj_row - 2))
        {
            test_idx = get_idx_fr_adj_colrow(adj_col + 1, adj_row - 2);
            if (!is_piece_same_color(piece, (ChessPiece)board[test_idx]))
            {
                out[mov_ctr++] = test_idx;
            }
        }

        if (is_adj_colrow_valid(adj_col - 1, adj_row + 2))
        {
            test_idx = get_idx_fr_adj_colrow(adj_col - 1, adj_row + 2);
            if (!is_piece_same_color(piece, (ChessPiece)board[test_idx]))
            {
                out[mov_ctr++] = test_idx;
            }
        }

        if (is_adj_colrow_valid(adj_col - 1, adj_row - 2))
        {
            test_idx = get_idx_fr_adj_colrow(adj_col - 1, adj_row - 2);
            if (!is_piece_same_color(piece, (ChessPiece)board[test_idx]))
            {
                out[mov_ctr++] = test_idx;
            }
        }

        if (is_adj_colrow_valid(adj_col + 2, adj_row + 1))
        {
            test_idx = get_idx_fr_adj_colrow(adj_col + 2, adj_row + 1);
            if (!is_piece_same_color(piece, (ChessPiece)board[test_idx]))
            {
                out[mov_ctr++] = test_idx;
            }
        }

        if (is_adj_colrow_valid(adj_col + 2, adj_row - 1))
        {
            test_idx = get_idx_fr_adj_colrow(adj_col + 2, adj_row - 1);
            if (!is_piece_same_color(piece, (ChessPiece)board[test_idx]))
            {
                out[mov_ctr++] = test_idx;
            }
        }

        if (is_adj_colrow_valid(adj_col - 2, adj_row + 1))
        {
            test_idx = get_idx_fr_adj_colrow(adj_col - 2, adj_row + 1);
            if (!is_piece_same_color(piece, (ChessPiece)board[test_idx]))
            {
                out[mov_ctr++] = test_idx;
            }
        }

        if (is_adj_colrow_valid(adj_col - 2, adj_row - 1))
        {
            test_idx = get_idx_fr_adj_colrow(adj_col - 2, adj_row - 1);
            if (!is_piece_same_color(piece, (ChessPiece)board[test_idx]))
            {
                out[mov_ctr++] = test_idx;
            }
        }
    }

    break;
    case WhiteBishop:
    case BlackBishop:
    {
        int ne = 0;
        int sw = 0;
        int se = 0;
        int nw = 0;
        for (int i = 1; i < 8; i++)
        {

            if (is_adj_colrow_valid(adj_col + i, adj_row + i) && ne == 0)
            {
                test_idx = get_idx_fr_adj_colrow(adj_col + i, adj_row + i);
                if (board[test_idx] != Empty)
                {
                    ne = 1;
                    if (!is_piece_same_color(piece, (ChessPiece)board[test_idx]))
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }
                else
                {
                    out[mov_ctr++] = test_idx;
                }
            }

            if (is_adj_colrow_valid(adj_col - i, adj_row - i) && sw == 0)
            {
                test_idx = get_idx_fr_adj_colrow(adj_col - i, adj_row - i);
                if (board[test_idx] != Empty)
                {
                    sw = 1;
                    if (!is_piece_same_color(piece, (ChessPiece)board[test_idx]))
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }
                else
                {
                    out[mov_ctr++] = test_idx;
                }
            }

            if (is_adj_colrow_valid(adj_col + i, adj_row - i) && se == 0)
            {
                test_idx = get_idx_fr_adj_colrow(adj_col + i, adj_row - i);
                if (board[test_idx] != Empty)
                {
                    se = 1;
                    if (!is_piece_same_color(piece, (ChessPiece)board[test_idx]))
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }
                else
                {
                    out[mov_ctr++] = test_idx;
                }
            }

            if (is_adj_colrow_valid(adj_col - i, adj_row + i) && nw == 0)
            {
                test_idx = get_idx_fr_adj_colrow(adj_col - i, adj_row + i);
                if (board[test_idx] != Empty)
                {
                    nw = 1;
                    if (!is_piece_same_color(piece, (ChessPiece)board[test_idx]))
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }
                else
                {
                    out[mov_ctr++] = test_idx;
                }
            }
        }
    }

    break;
    case WhiteRook:
    case BlackRook:
    {
        int n = 0;
        int s = 0;
        int e = 0;
        int w = 0;
        for (int i = 1; i < 8; i++)
        {

            if (is_adj_colrow_valid(adj_col + i, adj_row) && e == 0)
            {
                test_idx = get_idx_fr_adj_colrow(adj_col + i, adj_row);
                if (board[test_idx] != Empty)
                {
                    e = 1;
                    if (!is_piece_same_color(piece, (ChessPiece)board[test_idx]))
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }
                else
                {
                    out[mov_ctr++] = test_idx;
                }
            }

            if (is_adj_colrow_valid(adj_col - i, adj_row) && w == 0)
            {
                test_idx = get_idx_fr_adj_colrow(adj_col - i, adj_row);
                if (board[test_idx] != Empty)
                {
                    w = 1;
                    if (!is_piece_same_color(piece, (ChessPiece)board[test_idx]))
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }
                else
                {
                    out[mov_ctr++] = test_idx;
                }
            }

            if (is_adj_colrow_valid(adj_col, adj_row + i) && n == 0)
            {
                test_idx = get_idx_fr_adj_colrow(adj_col, adj_row + i);
                if (board[test_idx] != Empty)
                {
                    n = 1;
                    if (!is_piece_same_color(piece, (ChessPiece)board[test_idx]))
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }
                else
                {
                    out[mov_ctr++] = test_idx;
                }
            }

            if (is_adj_colrow_valid(adj_col, adj_row - i) && s == 0)
            {
                test_idx = get_idx_fr_adj_colrow(adj_col, adj_row - i);
                if (board[test_idx] != Empty)
                {
                    s = 1;
                    if (!is_piece_same_color(piece, (ChessPiece)board[test_idx]))
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }
                else
                {
                    out[mov_ctr++] = test_idx;
                }
            }
        }
    }

    break;
    case WhiteQueen:
    case BlackQueen:
        // ne,sw,se,nw
        {
            int ne = 0;
            int sw = 0;
            int se = 0;
            int nw = 0;
            for (int i = 1; i < 8; i++)
            {

                if (is_adj_colrow_valid(adj_col + i, adj_row + i) && ne == 0)
                {
                    test_idx = get_idx_fr_adj_colrow(adj_col + i, adj_row + i);
                    if (board[test_idx] != Empty)
                    {
                        ne = 1;
                        if (!is_piece_same_color(piece, (ChessPiece)board[test_idx]))
                        {
                            out[mov_ctr++] = test_idx;
                        }
                    }
                    else
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }

                if (is_adj_colrow_valid(adj_col - i, adj_row - i) && sw == 0)
                {
                    test_idx = get_idx_fr_adj_colrow(adj_col - i, adj_row - i);
                    if (board[test_idx] != Empty)
                    {
                        sw = 1;
                        if (!is_piece_same_color(piece, (ChessPiece)board[test_idx]))
                        {
                            out[mov_ctr++] = test_idx;
                        }
                    }
                    else
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }

                if (is_adj_colrow_valid(adj_col + i, adj_row - i) && se == 0)
                {
                    test_idx = get_idx_fr_adj_colrow(adj_col + i, adj_row - i);
                    if (board[test_idx] != Empty)
                    {
                        se = 1;
                        if (!is_piece_same_color(piece, (ChessPiece)board[test_idx]))
                        {
                            out[mov_ctr++] = test_idx;
                        }
                    }
                    else
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }

                if (is_adj_colrow_valid(adj_col - i, adj_row + i) && nw == 0)
                {
                    test_idx = get_idx_fr_adj_colrow(adj_col - i, adj_row + i);
                    if (board[test_idx] != Empty)
                    {
                        nw = 1;
                        if (!is_piece_same_color(piece, (ChessPiece)board[test_idx]))
                        {
                            out[mov_ctr++] = test_idx;
                        }
                    }
                    else
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }
            }
        }
        // n,s,e,w
        {
            int n = 0;
            int s = 0;
            int e = 0;
            int w = 0;
            for (int i = 1; i < 8; i++)
            {

                if (is_adj_colrow_valid(adj_col + i, adj_row) && e == 0)
                {
                    test_idx = get_idx_fr_adj_colrow(adj_col + i, adj_row);
                    if (board[test_idx] != Empty)
                    {
                        e = 1;
                        if (!is_piece_same_color(piece, (ChessPiece)board[test_idx]))
                        {
                            out[mov_ctr++] = test_idx;
                        }
                    }
                    else
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }

                if (is_adj_colrow_valid(adj_col - i, adj_row) && w == 0)
                {
                    test_idx = get_idx_fr_adj_colrow(adj_col - i, adj_row);
                    if (board[test_idx] != Empty)
                    {
                        w = 1;
                        if (!is_piece_same_color(piece, (ChessPiece)board[test_idx]))
                        {
                            out[mov_ctr++] = test_idx;
                        }
                    }
                    else
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }

                if (is_adj_colrow_valid(adj_col, adj_row + i) && n == 0)
                {
                    test_idx = get_idx_fr_adj_colrow(adj_col, adj_row + i);
                    if (board[test_idx] != Empty)
                    {
                        n = 1;
                        if (!is_piece_same_color(piece, (ChessPiece)board[test_idx]))
                        {
                            out[mov_ctr++] = test_idx;
                        }
                    }
                    else
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }

                if (is_adj_colrow_valid(adj_col, adj_row - i) && s == 0)
                {
                    test_idx = get_idx_fr_adj_colrow(adj_col, adj_row - i);
                    if (board[test_idx] != Empty)
                    {
                        s = 1;
                        if (!is_piece_same_color(piece, (ChessPiece)board[test_idx]))
                        {
                            out[mov_ctr++] = test_idx;
                        }
                    }
                    else
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }
            }
        }

        break;
    case WhiteKing:
    case BlackKing:
        // ne,sw,se,nw
        {
            int ne = 0;
            int sw = 0;
            int se = 0;
            int nw = 0;
            for (int i = 1; i < 2; i++)
            {

                if (is_adj_colrow_valid(adj_col + i, adj_row + i) && ne == 0)
                {
                    test_idx = get_idx_fr_adj_colrow(adj_col + i, adj_row + i);
                    if (board[test_idx] != Empty)
                    {
                        ne = 1;
                        if (!is_piece_same_color(piece, (ChessPiece)board[test_idx]))
                        {
                            out[mov_ctr++] = test_idx;
                        }
                    }
                    else
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }

                if (is_adj_colrow_valid(adj_col - i, adj_row - i) && sw == 0)
                {
                    test_idx = get_idx_fr_adj_colrow(adj_col - i, adj_row - i);
                    if (board[test_idx] != Empty)
                    {
                        sw = 1;
                        if (!is_piece_same_color(piece, (ChessPiece)board[test_idx]))
                        {
                            out[mov_ctr++] = test_idx;
                        }
                    }
                    else
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }

                if (is_adj_colrow_valid(adj_col + i, adj_row - i) && se == 0)
                {
                    test_idx = get_idx_fr_adj_colrow(adj_col + i, adj_row - i);
                    if (board[test_idx] != Empty)
                    {
                        se = 1;
                        if (!is_piece_same_color(piece, (ChessPiece)board[test_idx]))
                        {
                            out[mov_ctr++] = test_idx;
                        }
                    }
                    else
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }

                if (is_adj_colrow_valid(adj_col - i, adj_row + i) && nw == 0)
                {
                    test_idx = get_idx_fr_adj_colrow(adj_col - i, adj_row + i);
                    if (board[test_idx] != Empty)
                    {
                        nw = 1;
                        if (!is_piece_same_color(piece, (ChessPiece)board[test_idx]))
                        {
                            out[mov_ctr++] = test_idx;
                        }
                    }
                    else
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }
            }
        }
        // n,s,e,w
        {
            int n = 0;
            int s = 0;
            int e = 0;
            int w = 0;
            for (int i = 1; i < 2; i++)
            {

                if (is_adj_colrow_valid(adj_col + i, adj_row) && e == 0)
                {
                    test_idx = get_idx_fr_adj_colrow(adj_col + i, adj_row);
                    if (board[test_idx] != Empty)
                    {
                        e = 1;
                        if (!is_piece_same_color(piece, (ChessPiece)board[test_idx]))
                        {
                            out[mov_ctr++] = test_idx;
                        }
                    }
                    else
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }

                if (is_adj_colrow_valid(adj_col - i, adj_row) && w == 0)
                {
                    test_idx = get_idx_fr_adj_colrow(adj_col - i, adj_row);
                    if (board[test_idx] != Empty)
                    {
                        w = 1;
                        if (!is_piece_same_color(piece, (ChessPiece)board[test_idx]))
                        {
                            out[mov_ctr++] = test_idx;
                        }
                    }
                    else
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }

                if (is_adj_colrow_valid(adj_col, adj_row + i) && n == 0)
                {
                    test_idx = get_idx_fr_adj_colrow(adj_col, adj_row + i);
                    if (board[test_idx] != Empty)
                    {
                        n = 1;
                        if (!is_piece_same_color(piece, (ChessPiece)board[test_idx]))
                        {
                            out[mov_ctr++] = test_idx;
                        }
                    }
                    else
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }

                if (is_adj_colrow_valid(adj_col, adj_row - i) && s == 0)
                {
                    test_idx = get_idx_fr_adj_colrow(adj_col, adj_row - i);
                    if (board[test_idx] != Empty)
                    {
                        s = 1;
                        if (!is_piece_same_color(piece, (ChessPiece)board[test_idx]))
                        {
                            out[mov_ctr++] = test_idx;
                        }
                    }
                    else
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }
            }
        }

        // Castles.
        if (piece == WhiteKing)
        {
            if (col == 'e' && row == 1)
            {
                // Queen side castle.
                if (board[get_idx_fr_colrow('a', 1)] == WhiteRook)
                {
                    if (board[get_idx_fr_colrow('b', 1)] == Empty && board[get_idx_fr_colrow('c', 1)] == Empty && board[get_idx_fr_colrow('d', 1)] == Empty &&
                        !is_cell_under_attack(board, get_idx_fr_colrow('b', 1), true) && !is_cell_under_attack(board, get_idx_fr_colrow('c', 1), true) && !is_cell_under_attack(board, get_idx_fr_colrow('d', 1), true))
                    {
                        out[mov_ctr++] = get_idx_fr_colrow('c', 1);
                    }
                }

                // King side castle.
                if (board[get_idx_fr_colrow('h', 1)] == WhiteRook)
                {
                    if (board[get_idx_fr_colrow('f', 1)] == Empty && board[get_idx_fr_colrow('g', 1)] == Empty &&
                        !is_cell_under_attack(board, get_idx_fr_colrow('f', 1), true) && !is_cell_under_attack(board, get_idx_fr_colrow('g', 1), true))
                    {
                        out[mov_ctr++] = get_idx_fr_colrow('g', 1);
                    }
                }
            }
        }
        else
        {
            if (col == 'e' && row == 8)
            {
                // Queen side castle.
                if (board[get_idx_fr_colrow('a', 8)] == BlackRook)
                {
                    if (board[get_idx_fr_colrow('b', 8)] == Empty && board[get_idx_fr_colrow('c', 8)] == Empty && board[get_idx_fr_colrow('d', 8)] == Empty &&
                        !is_cell_under_attack(board, get_idx_fr_colrow('b', 8), false) && !is_cell_under_attack(board, get_idx_fr_colrow('c', 8), false) && !is_cell_under_attack(board, get_idx_fr_colrow('d', 8), false))
                    {
                        out[mov_ctr++] = get_idx_fr_colrow('c', 8);
                    }
                }

                // King side castle.
                if (board[get_idx_fr_colrow('h', 8)] == BlackRook)
                {
                    if (board[get_idx_fr_colrow('f', 8)] == Empty && board[get_idx_fr_colrow('g', 8)] == Empty &&
                        !is_cell_under_attack(board, get_idx_fr_colrow('f', 8), false) && !is_cell_under_attack(board, get_idx_fr_colrow('g', 8), false))
                    {
                        out[mov_ctr++] = get_idx_fr_colrow('g', 8);
                    }
                }
            }
        }

        break;
    default: // Nothing...
        break;
    }

    if (test_in_check_flg)
    {
        int check_out[CHESS_MAX_LEGAL_MOVE_CNT];
        memset(check_out, CHESS_INVALID_VALUE, sizeof(int) * CHESS_MAX_LEGAL_MOVE_CNT);
        int check_mov_ctr = 0;
        int cpy_board[CHESS_BOARD_LEN];
        for (int i = 0; i < mov_ctr; i++)
        {
            simulate_board_change_w_srcdst_idx(board, piece_idx, out[i], cpy_board);
            if (!is_in_check(cpy_board, white_mov_flg))
            {
                check_out[check_mov_ctr++] = out[i];
            }
        }

        memcpy(out, check_out, sizeof(int) * mov_ctr);
    }
}

void get_piece_influence(int *board, int piece_idx, int *out)
{
    memset(out, CHESS_INVALID_VALUE, sizeof(int) * CHESS_MAX_LEGAL_MOVE_CNT);
    int mov_ctr = 0;

    ChessPiece piece = (ChessPiece)board[piece_idx];

    char col = get_col_fr_idx(piece_idx);
    int row = get_row_fr_idx(piece_idx);

    int adj_col = get_adj_col_fr_idx(piece_idx);
    int adj_row = get_adj_row_fr_idx(piece_idx);

    int white_mov_flg;
    if (is_piece_white(piece))
    {
        white_mov_flg = 1;
    }
    else
    {
        white_mov_flg = 0;
    }

    int test_idx;

    switch (piece)
    {
    case WhitePawn:
        // TODO: au passant
        {
            test_idx = get_idx_fr_colrow(col, row + 1);
            if (is_row_valid(row + 1) && board[test_idx] == Empty)
            {
                out[mov_ctr++] = test_idx;
            }

            test_idx = get_idx_fr_adj_colrow(adj_col - 1, adj_row + 1);
            if (is_adj_colrow_valid(adj_col - 1, adj_row + 1))
            {
                out[mov_ctr++] = test_idx;
            }

            test_idx = get_idx_fr_adj_colrow(adj_col + 1, adj_row + 1);
            if (is_adj_colrow_valid(adj_col + 1, adj_row + 1))
            {
                out[mov_ctr++] = test_idx;
            }

            if (row == 2)
            {
                // Dont need to check if row adjustments are valid since we know that starting row is 2.
                test_idx = get_idx_fr_colrow(col, row + 1);
                if (board[test_idx] == Empty)
                {
                    test_idx = get_idx_fr_colrow(col, row + 2);
                    if (board[test_idx] == Empty)
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }
            }
        }

        break;
    case BlackPawn:
        // TODO: au passant
        {
            test_idx = get_idx_fr_colrow(col, row - 1);
            if (is_row_valid(row - 1) && board[test_idx] == Empty)
            {
                out[mov_ctr++] = test_idx;
            }

            test_idx = get_idx_fr_adj_colrow(adj_col - 1, adj_row - 1);
            if (is_adj_colrow_valid(adj_col - 1, adj_row - 1))
            {
                out[mov_ctr++] = test_idx;
            }

            test_idx = get_idx_fr_adj_colrow(adj_col + 1, adj_row - 1);
            if (is_adj_colrow_valid(adj_col + 1, adj_row - 1))
            {
                out[mov_ctr++] = test_idx;
            }

            if (row == 7)
            {
                // Dont need to check if row adjustments are valid since we know that starting row is 7.
                test_idx = get_idx_fr_colrow(col, row - 1);
                if (board[test_idx] == Empty)
                {
                    test_idx = get_idx_fr_colrow(col, row - 2);
                    if (board[test_idx] == Empty)
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }
            }
        }

        break;
    case WhiteKnight:
    case BlackKnight:
    {

        if (is_adj_colrow_valid(adj_col + 1, adj_row + 2))
        {
            test_idx = get_idx_fr_adj_colrow(adj_col + 1, adj_row + 2);
            out[mov_ctr++] = test_idx;
        }

        if (is_adj_colrow_valid(adj_col + 1, adj_row - 2))
        {
            test_idx = get_idx_fr_adj_colrow(adj_col + 1, adj_row - 2);
            out[mov_ctr++] = test_idx;
        }

        if (is_adj_colrow_valid(adj_col - 1, adj_row + 2))
        {
            test_idx = get_idx_fr_adj_colrow(adj_col - 1, adj_row + 2);
            out[mov_ctr++] = test_idx;
        }

        if (is_adj_colrow_valid(adj_col - 1, adj_row - 2))
        {
            test_idx = get_idx_fr_adj_colrow(adj_col - 1, adj_row - 2);
            out[mov_ctr++] = test_idx;
        }

        if (is_adj_colrow_valid(adj_col + 2, adj_row + 1))
        {
            test_idx = get_idx_fr_adj_colrow(adj_col + 2, adj_row + 1);
            out[mov_ctr++] = test_idx;
        }

        if (is_adj_colrow_valid(adj_col + 2, adj_row - 1))
        {
            test_idx = get_idx_fr_adj_colrow(adj_col + 2, adj_row - 1);
            out[mov_ctr++] = test_idx;
        }

        if (is_adj_colrow_valid(adj_col - 2, adj_row + 1))
        {
            test_idx = get_idx_fr_adj_colrow(adj_col - 2, adj_row + 1);
            out[mov_ctr++] = test_idx;
        }

        if (is_adj_colrow_valid(adj_col - 2, adj_row - 1))
        {
            test_idx = get_idx_fr_adj_colrow(adj_col - 2, adj_row - 1);
            out[mov_ctr++] = test_idx;
        }
    }

    break;
    case WhiteBishop:
    case BlackBishop:
    {
        int ne = 0;
        int sw = 0;
        int se = 0;
        int nw = 0;
        for (int i = 1; i < 8; i++)
        {

            if (is_adj_colrow_valid(adj_col + i, adj_row + i) && ne == 0)
            {
                test_idx = get_idx_fr_adj_colrow(adj_col + i, adj_row + i);
                if (board[test_idx] != Empty)
                {
                    ne = 1;
                    out[mov_ctr++] = test_idx;
                }
                else
                {
                    out[mov_ctr++] = test_idx;
                }
            }

            if (is_adj_colrow_valid(adj_col - i, adj_row - i) && sw == 0)
            {
                test_idx = get_idx_fr_adj_colrow(adj_col - i, adj_row - i);
                if (board[test_idx] != Empty)
                {
                    sw = 1;
                    out[mov_ctr++] = test_idx;
                }
                else
                {
                    out[mov_ctr++] = test_idx;
                }
            }

            if (is_adj_colrow_valid(adj_col + i, adj_row - i) && se == 0)
            {
                test_idx = get_idx_fr_adj_colrow(adj_col + i, adj_row - i);
                if (board[test_idx] != Empty)
                {
                    se = 1;
                    out[mov_ctr++] = test_idx;
                }
                else
                {
                    out[mov_ctr++] = test_idx;
                }
            }

            if (is_adj_colrow_valid(adj_col - i, adj_row + i) && nw == 0)
            {
                test_idx = get_idx_fr_adj_colrow(adj_col - i, adj_row + i);
                if (board[test_idx] != Empty)
                {
                    nw = 1;
                    out[mov_ctr++] = test_idx;
                }
                else
                {
                    out[mov_ctr++] = test_idx;
                }
            }
        }
    }

    break;
    case WhiteRook:
    case BlackRook:
    {
        int n = 0;
        int s = 0;
        int e = 0;
        int w = 0;
        for (int i = 1; i < 8; i++)
        {

            if (is_adj_colrow_valid(adj_col + i, adj_row) && e == 0)
            {
                test_idx = get_idx_fr_adj_colrow(adj_col + i, adj_row);
                if (board[test_idx] != Empty)
                {
                    e = 1;
                    out[mov_ctr++] = test_idx;
                }
                else
                {
                    out[mov_ctr++] = test_idx;
                }
            }

            if (is_adj_colrow_valid(adj_col - i, adj_row) && w == 0)
            {
                test_idx = get_idx_fr_adj_colrow(adj_col - i, adj_row);
                if (board[test_idx] != Empty)
                {
                    w = 1;
                    out[mov_ctr++] = test_idx;
                }
                else
                {
                    out[mov_ctr++] = test_idx;
                }
            }

            if (is_adj_colrow_valid(adj_col, adj_row + i) && n == 0)
            {
                test_idx = get_idx_fr_adj_colrow(adj_col, adj_row + i);
                if (board[test_idx] != Empty)
                {
                    n = 1;
                    out[mov_ctr++] = test_idx;
                }
                else
                {
                    out[mov_ctr++] = test_idx;
                }
            }

            if (is_adj_colrow_valid(adj_col, adj_row - i) && s == 0)
            {
                test_idx = get_idx_fr_adj_colrow(adj_col, adj_row - i);
                if (board[test_idx] != Empty)
                {
                    s = 1;
                    out[mov_ctr++] = test_idx;
                }
                else
                {
                    out[mov_ctr++] = test_idx;
                }
            }
        }
    }

    break;
    case WhiteQueen:
    case BlackQueen:
        // ne,sw,se,nw
        {
            int ne = 0;
            int sw = 0;
            int se = 0;
            int nw = 0;
            for (int i = 1; i < 8; i++)
            {

                if (is_adj_colrow_valid(adj_col + i, adj_row + i) && ne == 0)
                {
                    test_idx = get_idx_fr_adj_colrow(adj_col + i, adj_row + i);
                    if (board[test_idx] != Empty)
                    {
                        ne = 1;
                        out[mov_ctr++] = test_idx;
                    }
                    else
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }

                if (is_adj_colrow_valid(adj_col - i, adj_row - i) && sw == 0)
                {
                    test_idx = get_idx_fr_adj_colrow(adj_col - i, adj_row - i);
                    if (board[test_idx] != Empty)
                    {
                        sw = 1;
                        out[mov_ctr++] = test_idx;
                    }
                    else
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }

                if (is_adj_colrow_valid(adj_col + i, adj_row - i) && se == 0)
                {
                    test_idx = get_idx_fr_adj_colrow(adj_col + i, adj_row - i);
                    if (board[test_idx] != Empty)
                    {
                        se = 1;
                        out[mov_ctr++] = test_idx;
                    }
                    else
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }

                if (is_adj_colrow_valid(adj_col - i, adj_row + i) && nw == 0)
                {
                    test_idx = get_idx_fr_adj_colrow(adj_col - i, adj_row + i);
                    if (board[test_idx] != Empty)
                    {
                        nw = 1;
                        out[mov_ctr++] = test_idx;
                    }
                    else
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }
            }
        }
        // n,s,e,w
        {
            int n = 0;
            int s = 0;
            int e = 0;
            int w = 0;
            for (int i = 1; i < 8; i++)
            {

                if (is_adj_colrow_valid(adj_col + i, adj_row) && e == 0)
                {
                    test_idx = get_idx_fr_adj_colrow(adj_col + i, adj_row);
                    if (board[test_idx] != Empty)
                    {
                        e = 1;
                        out[mov_ctr++] = test_idx;
                    }
                    else
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }

                if (is_adj_colrow_valid(adj_col - i, adj_row) && w == 0)
                {
                    test_idx = get_idx_fr_adj_colrow(adj_col - i, adj_row);
                    if (board[test_idx] != Empty)
                    {
                        w = 1;
                        out[mov_ctr++] = test_idx;
                    }
                    else
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }

                if (is_adj_colrow_valid(adj_col, adj_row + i) && n == 0)
                {
                    test_idx = get_idx_fr_adj_colrow(adj_col, adj_row + i);
                    if (board[test_idx] != Empty)
                    {
                        n = 1;
                        out[mov_ctr++] = test_idx;
                    }
                    else
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }

                if (is_adj_colrow_valid(adj_col, adj_row - i) && s == 0)
                {
                    test_idx = get_idx_fr_adj_colrow(adj_col, adj_row - i);
                    if (board[test_idx] != Empty)
                    {
                        s = 1;
                        out[mov_ctr++] = test_idx;
                    }
                    else
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }
            }
        }

        break;
    case WhiteKing:
    case BlackKing:
        // ne,sw,se,nw
        {
            int ne = 0;
            int sw = 0;
            int se = 0;
            int nw = 0;
            for (int i = 1; i < 2; i++)
            {

                if (is_adj_colrow_valid(adj_col + i, adj_row + i) && ne == 0)
                {
                    test_idx = get_idx_fr_adj_colrow(adj_col + i, adj_row + i);
                    if (board[test_idx] != Empty)
                    {
                        ne = 1;
                        out[mov_ctr++] = test_idx;
                    }
                    else
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }

                if (is_adj_colrow_valid(adj_col - i, adj_row - i) && sw == 0)
                {
                    test_idx = get_idx_fr_adj_colrow(adj_col - i, adj_row - i);
                    if (board[test_idx] != Empty)
                    {
                        sw = 1;
                        out[mov_ctr++] = test_idx;
                    }
                    else
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }

                if (is_adj_colrow_valid(adj_col + i, adj_row - i) && se == 0)
                {
                    test_idx = get_idx_fr_adj_colrow(adj_col + i, adj_row - i);
                    if (board[test_idx] != Empty)
                    {
                        se = 1;
                        out[mov_ctr++] = test_idx;
                    }
                    else
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }

                if (is_adj_colrow_valid(adj_col - i, adj_row + i) && nw == 0)
                {
                    test_idx = get_idx_fr_adj_colrow(adj_col - i, adj_row + i);
                    if (board[test_idx] != Empty)
                    {
                        nw = 1;
                        out[mov_ctr++] = test_idx;
                    }
                    else
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }
            }
        }
        // n,s,e,w
        {
            int n = 0;
            int s = 0;
            int e = 0;
            int w = 0;
            for (int i = 1; i < 2; i++)
            {

                if (is_adj_colrow_valid(adj_col + i, adj_row) && e == 0)
                {
                    test_idx = get_idx_fr_adj_colrow(adj_col + i, adj_row);
                    if (board[test_idx] != Empty)
                    {
                        e = 1;
                        out[mov_ctr++] = test_idx;
                    }
                    else
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }

                if (is_adj_colrow_valid(adj_col - i, adj_row) && w == 0)
                {
                    test_idx = get_idx_fr_adj_colrow(adj_col - i, adj_row);
                    if (board[test_idx] != Empty)
                    {
                        w = 1;
                        out[mov_ctr++] = test_idx;
                    }
                    else
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }

                if (is_adj_colrow_valid(adj_col, adj_row + i) && n == 0)
                {
                    test_idx = get_idx_fr_adj_colrow(adj_col, adj_row + i);
                    if (board[test_idx] != Empty)
                    {
                        n = 1;
                        out[mov_ctr++] = test_idx;
                    }
                    else
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }

                if (is_adj_colrow_valid(adj_col, adj_row - i) && s == 0)
                {
                    test_idx = get_idx_fr_adj_colrow(adj_col, adj_row - i);
                    if (board[test_idx] != Empty)
                    {
                        s = 1;
                        out[mov_ctr++] = test_idx;
                    }
                    else
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }
            }
        }

        // Castles.
        if (piece == WhiteKing)
        {
            if (col == 'e' && row == 1)
            {
                // Queen side castle.
                if (board[get_idx_fr_colrow('a', 1)] == WhiteRook)
                {
                    if (board[get_idx_fr_colrow('b', 1)] == Empty && board[get_idx_fr_colrow('c', 1)] == Empty && board[get_idx_fr_colrow('d', 1)] == Empty &&
                        !is_cell_under_attack(board, get_idx_fr_colrow('b', 1), true) && !is_cell_under_attack(board, get_idx_fr_colrow('c', 1), true) && !is_cell_under_attack(board, get_idx_fr_colrow('d', 1), true))
                    {
                        out[mov_ctr++] = get_idx_fr_colrow('c', 1);
                    }
                }

                // King side castle.
                if (board[get_idx_fr_colrow('h', 1)] == WhiteRook)
                {
                    if (board[get_idx_fr_colrow('f', 1)] == Empty && board[get_idx_fr_colrow('g', 1)] == Empty &&
                        !is_cell_under_attack(board, get_idx_fr_colrow('f', 1), true) && !is_cell_under_attack(board, get_idx_fr_colrow('g', 1), true))
                    {
                        out[mov_ctr++] = get_idx_fr_colrow('g', 1);
                    }
                }
            }
        }
        else
        {
            if (col == 'e' && row == 8)
            {
                // Queen side castle.
                if (board[get_idx_fr_colrow('a', 8)] == BlackRook)
                {
                    if (board[get_idx_fr_colrow('b', 8)] == Empty && board[get_idx_fr_colrow('c', 8)] == Empty && board[get_idx_fr_colrow('d', 8)] == Empty &&
                        !is_cell_under_attack(board, get_idx_fr_colrow('b', 8), false) && !is_cell_under_attack(board, get_idx_fr_colrow('c', 8), false) && !is_cell_under_attack(board, get_idx_fr_colrow('d', 8), false))
                    {
                        out[mov_ctr++] = get_idx_fr_colrow('c', 8);
                    }
                }

                // King side castle.
                if (board[get_idx_fr_colrow('h', 8)] == BlackRook)
                {
                    if (board[get_idx_fr_colrow('f', 8)] == Empty && board[get_idx_fr_colrow('g', 8)] == Empty &&
                        !is_cell_under_attack(board, get_idx_fr_colrow('f', 8), false) && !is_cell_under_attack(board, get_idx_fr_colrow('g', 8), false))
                    {
                        out[mov_ctr++] = get_idx_fr_colrow('g', 8);
                    }
                }
            }
        }

        break;
    default: // Nothing...
        break;
    }

    // Test in check:
    {
        int check_out[CHESS_MAX_LEGAL_MOVE_CNT];
        memset(check_out, CHESS_INVALID_VALUE, sizeof(int) * CHESS_MAX_LEGAL_MOVE_CNT);
        int check_mov_ctr = 0;
        int cpy_board[CHESS_BOARD_LEN];
        for (int i = 0; i < mov_ctr; i++)
        {
            simulate_board_change_w_srcdst_idx(board, piece_idx, out[i], cpy_board);
            if (!is_in_check(cpy_board, white_mov_flg))
            {
                check_out[check_mov_ctr++] = out[i];
            }
        }

        memcpy(out, check_out, sizeof(int) * mov_ctr);
    }
}

SrcDst_Idx get_random_move(int *board, bool white_mov_flg, int *cmp_board)
{
    int sim_board[CHESS_BOARD_LEN];
    int legal_moves[CHESS_MAX_LEGAL_MOVE_CNT];
    int piece_idxs[CHESS_BOARD_LEN];

    // Get piece indexes.

    int piece_ctr = 0;
    for (int i = 0; i < CHESS_BOARD_LEN; i++)
    {
        if (white_mov_flg)
        {
            if (is_piece_white((ChessPiece)board[i]))
            {
                piece_idxs[piece_ctr++] = i;
            }
        }
        else
        {
            if (is_piece_black((ChessPiece)board[i]) == 1)
            {
                piece_idxs[piece_ctr++] = i;
            }
        }
    }

    SrcDst_Idx src_dst_idx;
    src_dst_idx.src_idx = CHESS_INVALID_VALUE;
    src_dst_idx.dst_idx = CHESS_INVALID_VALUE;
    int max_try_cnt = 20;
    int try_ctr = 0;
    while (try_ctr < max_try_cnt)
    {
        int rand_piece_idx = rand() % piece_ctr;

        // Got our piece; now get moves.
        int legal_mov_ctr = 0;
        get_legal_moves(board, piece_idxs[rand_piece_idx], legal_moves, 1);
        for (int i = 0; i < CHESS_MAX_LEGAL_MOVE_CNT; i++)
        {
            if (legal_moves[i] == CHESS_INVALID_VALUE)
            {
                break;
            }
            else
            {
                legal_mov_ctr++;
            }
        }

        // If at least 1 move found, randomly make one and compare.
        if (legal_mov_ctr > 0)
        {
            int rand_legal_mov_idx = rand() % legal_mov_ctr;
            simulate_board_change_w_srcdst_idx(board, piece_idxs[rand_piece_idx],
                                               legal_moves[rand_legal_mov_idx], sim_board);

            // Make sure the same move was not made.
            if (boardcmp(cmp_board, sim_board) != 0)
            {
                src_dst_idx.src_idx = piece_idxs[rand_piece_idx];
                src_dst_idx.dst_idx = legal_moves[rand_legal_mov_idx];
                break;
            }
        }

        try_ctr++;
    }

    return src_dst_idx;
}

void simulate_board_change_w_srcdst_idx(int *board, int src_idx, int dst_idx, int *out)
{
    copy_board(board, out);
    out[dst_idx] = out[src_idx];
    out[src_idx] = Empty;
}

void translate_srcdst_idx_to_mov(int *board, int src_idx, int dst_idx, char *out)
{
    memset(out, 0, CHESS_MAX_MOVE_LEN);

    ChessPiece piece = (ChessPiece)board[src_idx];
    char piece_id = get_char_fr_piece((ChessPiece)board[src_idx]);
    char src_col = get_col_fr_idx(src_idx);
    int src_row = get_row_fr_idx(src_idx);
    char dst_col = get_col_fr_idx(dst_idx);
    int dst_row = get_row_fr_idx(dst_idx);

    // Check for castle.
    if (piece == WhiteKing || piece == BlackKing)
    {
        int src_adj_col = get_adj_col_fr_col(src_col);
        int src_adj_row = get_adj_row_fr_row(src_row);
        int dst_adj_col = get_adj_col_fr_col(dst_col);
        int dst_adj_row = get_adj_row_fr_row(dst_row);
        if ((src_adj_col - dst_adj_col) == -2)
        {
            memcpy(out, "O-O", 3);
            return;
        }
        else if ((src_adj_col - dst_adj_col) == 2)
        {
            memcpy(out, "O-O-O", 5);
            return;
        }
    }

    // Example format going forward: piece id|src col|src row|dst col|dst row|promo (or space)|promo piece id (or space)
    // ^always 7 chars

    int move_ctr = 0;

    out[move_ctr++] = piece_id;

    out[move_ctr++] = src_col;
    out[move_ctr++] = (char)(src_row + '0');

    out[move_ctr++] = dst_col;
    out[move_ctr++] = (char)(dst_row + '0');

    // Check for pawn promotion. If none, set last 2 chars to ' '.
    if ((piece == WhitePawn && dst_row == 8) || (piece == BlackPawn && dst_row == 1))
    {
        out[move_ctr++] = '=';
        out[move_ctr++] = 'Q';
    }
    else
    {
        out[move_ctr++] = ' ';
        out[move_ctr++] = ' ';
    }
}

void change_board_w_mov(int *board, const char *immut_mov, bool white_mov_flg)
{
    char mut_mov[CHESS_MAX_MOVE_LEN];
    memcpy(mut_mov, immut_mov, CHESS_MAX_MOVE_LEN);

    int src_idx;
    int dst_idx;
    char src_col;
    char dst_col;
    int src_row;
    int dst_row;
    ChessPiece piece;
    char piece_char;

    int legal_moves[CHESS_MAX_LEGAL_MOVE_CNT];

    // Trim '+'/'#'.
    for (int i = CHESS_MAX_MOVE_LEN; i > 0; i--)
    {
        if (mut_mov[i] == '+' || mut_mov[i] == '#')
        {
            // Can safely just 0 out since we know '+'/'#' will be at the end of the move string.
            mut_mov[i] = 0;
        }
    }

    // Remove 'x'.
    for (int i = 0; i < CHESS_MAX_MOVE_LEN; i++)
    {
        if (mut_mov[i] == 'x')
        {
            for (int j = i; j < CHESS_MAX_MOVE_LEN - 1; j++)
            {
                mut_mov[j] = mut_mov[j + 1];
            }
            break;
        }
    }

    int mut_mov_len = strlen(mut_mov);

    switch (mut_mov_len)
    {
    case 2:
        // Pawn move.
        dst_row = get_row_fr_char(mut_mov[1]);
        dst_idx = get_idx_fr_colrow(mut_mov[0], dst_row);
        if (white_mov_flg)
        {
            int prev_idx = get_idx_fr_colrow(mut_mov[0], dst_row - 1);
            int prev_idx_2 = get_idx_fr_colrow(mut_mov[0], dst_row - 2);

            board[dst_idx] = WhitePawn;

            if (board[prev_idx] == WhitePawn)
            {
                board[prev_idx] = Empty;
            }
            else if (board[prev_idx_2] == WhitePawn)
            {
                board[prev_idx_2] = Empty;
            }
        }
        else
        {
            int prev_idx = get_idx_fr_colrow(mut_mov[0], dst_row + 1);
            int prev_idx_2 = get_idx_fr_colrow(mut_mov[0], dst_row + 2);

            board[dst_idx] = BlackPawn;

            if (board[prev_idx] == BlackPawn)
            {
                board[prev_idx] = Empty;
            }
            else if (board[prev_idx_2] == BlackPawn)
            {
                board[prev_idx_2] = Empty;
            }
        }
        break;
    case 3:
        if (strcmp(mut_mov, "O-O") == 0)
        {
            // King side castle.
            if (white_mov_flg)
            {
                int cur_rook_idx = get_idx_fr_colrow('h', 1);
                int cur_king_idx = get_idx_fr_colrow('e', 1);
                board[cur_rook_idx] = Empty;
                board[cur_king_idx] = Empty;

                int nxt_rook_idx = get_idx_fr_colrow('f', 1);
                int nxt_king_idx = get_idx_fr_colrow('g', 1);
                board[nxt_rook_idx] = WhiteRook;
                board[nxt_king_idx] = WhiteKing;
            }
            else
            {
                int cur_rook_idx = get_idx_fr_colrow('h', 8);
                int cur_king_idx = get_idx_fr_colrow('e', 8);
                board[cur_rook_idx] = Empty;
                board[cur_king_idx] = Empty;

                int nxt_rook_idx = get_idx_fr_colrow('f', 8);
                int nxt_king_idx = get_idx_fr_colrow('g', 8);
                board[nxt_rook_idx] = BlackRook;
                board[nxt_king_idx] = BlackKing;
            }
        }
        else
        {
            // Need to check if isupper since pawn move will not have piece id -- just the src col.
            if (isupper(mut_mov[0]) == 1)
            {
                // Minor/major piece move.
                piece = get_piece_fr_char(mut_mov[0], white_mov_flg);
                dst_row = get_row_fr_char(mut_mov[2]);
                dst_idx = get_idx_fr_colrow(mut_mov[1], dst_row);

                int found = 0;
                for (int i = 0; i < CHESS_BOARD_LEN; i++)
                {
                    if (board[i] == piece)
                    {
                        get_legal_moves(board, i, legal_moves, false);
                        for (int j = 0; j < CHESS_MAX_LEGAL_MOVE_CNT; j++)
                        {
                            if (legal_moves[j] == dst_idx)
                            {
                                board[dst_idx] = piece;
                                board[i] = Empty;
                                found = 1;
                                break;
                            }
                            else if (legal_moves[j] == CHESS_INVALID_VALUE)
                            {
                                break;
                            }
                        }
                    }
                    if (found == 1)
                    {
                        break;
                    }
                }
            }
            else
            {
                // Disambiguated pawn move.
                src_col = mut_mov[0];
                dst_row = get_row_fr_char(mut_mov[2]);
                dst_idx = get_idx_fr_colrow(mut_mov[1], dst_row);

                if (white_mov_flg)
                {
                    piece = WhitePawn;
                    board[get_idx_fr_colrow(src_col, dst_row - 1)] = Empty;
                }
                else
                {
                    piece = BlackPawn;
                    board[get_idx_fr_colrow(src_col, dst_row + 1)] = Empty;
                }

                board[dst_idx] = piece;
            }
        }
        break;
    case 4:
        // Need to check if isupper since pawn move will not have piece id -- just the src col.
        if (isupper(mut_mov[0]) == 1)
        {
            // Disambiguated minor/major piece move.
            dst_row = get_row_fr_char(mut_mov[3]);
            piece = get_piece_fr_char(mut_mov[0], white_mov_flg);
            src_col = mut_mov[1];
            dst_idx = get_idx_fr_colrow(mut_mov[2], dst_row);

            for (int i = 0; i < CHESS_BOARD_LEN; i++)
            {
                if (get_col_fr_idx(i) == src_col && board[i] == piece)
                {
                    board[dst_idx] = piece;
                    board[i] = Empty;
                    break;
                }
            }
        }
        else
        {
            // Pawn promotion.
            if (mut_mov[2] == '=')
            {
                dst_row = get_row_fr_char(mut_mov[1]);
                dst_idx = get_idx_fr_colrow(mut_mov[0], dst_row);
                piece_char = mut_mov[3];
                piece = get_piece_fr_char(piece_char, white_mov_flg);
                ChessPiece promo_piece = piece;

                if (white_mov_flg)
                {
                    piece = WhitePawn;
                }
                else
                {
                    piece = BlackPawn;
                }

                int found = 0;
                for (int i = 0; i < CHESS_BOARD_LEN; i++)
                {
                    if (board[i] == piece)
                    {
                        get_legal_moves(board, i, legal_moves, false);
                        for (int j = 0; j < CHESS_MAX_LEGAL_MOVE_CNT; j++)
                        {
                            if (legal_moves[j] == dst_idx)
                            {
                                board[i] = Empty;
                                found = 1;
                                break;
                            }
                            else if (legal_moves[j] == CHESS_INVALID_VALUE)
                            {
                                break;
                            }
                        }
                    }
                    if (found == 1)
                    {
                        break;
                    }
                }

                board[dst_idx] = promo_piece;
            }
        }
        break;
    case 5:
        if (strcmp(mut_mov, "O-O-O") == 0)
        {
            // Queen side castle.
            if (white_mov_flg)
            {
                int cur_rook_idx = get_idx_fr_colrow('a', 1);
                int cur_king_idx = get_idx_fr_colrow('e', 1);
                board[cur_rook_idx] = Empty;
                board[cur_king_idx] = Empty;

                int nxt_rook_idx = get_idx_fr_colrow('d', 1);
                int nxt_king_idx = get_idx_fr_colrow('c', 1);
                board[nxt_rook_idx] = WhiteRook;
                board[nxt_king_idx] = WhiteKing;
            }
            else
            {
                int cur_rook_idx = get_idx_fr_colrow('a', 8);
                int cur_king_idx = get_idx_fr_colrow('e', 8);
                board[cur_rook_idx] = Empty;
                board[cur_king_idx] = Empty;

                int nxt_rook_idx = get_idx_fr_colrow('d', 8);
                int nxt_king_idx = get_idx_fr_colrow('c', 8);
                board[nxt_rook_idx] = BlackRook;
                board[nxt_king_idx] = BlackKing;
            }
        }
        else
        {
            // Need to check if isupper since pawn move will not have piece id -- just the src col.
            if (isupper(mut_mov[0]) == 1)
            {
                // Disambiguated queen move.
                piece = get_piece_fr_char(mut_mov[0], white_mov_flg);
                if (piece == WhiteQueen || piece == BlackQueen)
                {
                    src_col = mut_mov[1];
                    src_row = get_row_fr_char(mut_mov[2]);
                    dst_col = mut_mov[3];
                    dst_row = get_row_fr_char(mut_mov[4]);

                    src_idx = get_idx_fr_colrow(src_col, src_row);
                    dst_idx = get_idx_fr_colrow(dst_col, dst_row);

                    board[dst_idx] = piece;
                    board[src_idx] = Empty;
                }
            }
            else
            {
                // Disambiguated pawn promotion.
                if (mut_mov[3] == '=')
                {
                    src_col = mut_mov[0];
                    dst_row = get_row_fr_char(mut_mov[2]);
                    dst_idx = get_idx_fr_colrow(mut_mov[1], dst_row);
                    piece_char = mut_mov[4];
                    piece = get_piece_fr_char(piece_char, white_mov_flg);
                    ChessPiece promo_piece = piece;

                    if (white_mov_flg)
                    {
                        piece = WhitePawn;
                    }
                    else
                    {
                        piece = BlackPawn;
                    }

                    for (int i = 0; i < CHESS_BOARD_LEN; i++)
                    {
                        if (get_col_fr_idx(i) == src_col && board[i] == piece)
                        {
                            board[dst_idx] = promo_piece;
                            board[i] = Empty;
                            break;
                        }
                    }
                }
            }
        }
        break;
    case 7: // 7 is chess-zero custom move format.
        if (mut_mov_len == 7)
        {
            piece = get_piece_fr_char(mut_mov[0], white_mov_flg);
            src_col = mut_mov[1];
            src_row = get_row_fr_char(mut_mov[2]);
            src_idx = get_idx_fr_colrow(src_col, src_row);
            dst_col = mut_mov[3];
            dst_row = get_row_fr_char(mut_mov[4]);
            dst_idx = get_idx_fr_colrow(dst_col, dst_row);

            if (mut_mov[5] == '=')
            {
                ChessPiece promo_piece = get_piece_fr_char(mut_mov[6], white_mov_flg);
                board[dst_idx] = promo_piece;
                board[src_idx] = Empty;
            }
            else
            {
                board[dst_idx] = piece;
                board[src_idx] = Empty;
            }
        }
        break;
    default: // Nothing..
        break;
    }
}

int boardcmp(int *a, int *b)
{
    return memcmp(a, b, sizeof(int) * CHESS_BOARD_LEN);
}

void print_board(int *board)
{
    // Print in a more viewable format(a8 at top left of screen).
    printf("   +---+---+---+---+---+---+---+---+");
    printf("\n");
    for (int i = CHESS_BOARD_ROW_CNT - 1; i >= 0; i--)
    {
        printf("%d  ", i + 1);
        printf("|");
        for (int j = 0; j < CHESS_BOARD_COL_CNT; j++)
        {

            switch (board[(i * CHESS_BOARD_COL_CNT) + j])
            {
            case WhitePawn:
                printf(" P |");
                break;
            case BlackPawn:
                printf(" p |");
                break;
            case WhiteKnight:
                printf(" N |");
                break;
            case BlackKnight:
                printf(" n |");
                break;
            case WhiteBishop:
                printf(" B |");
                break;
            case BlackBishop:
                printf(" b |");
                break;
            case WhiteRook:
                printf(" R |");
                break;
            case BlackRook:
                printf(" r |");
                break;
            case WhiteQueen:
                printf(" Q |");
                break;
            case BlackQueen:
                printf(" q |");
                break;
            case WhiteKing:
                printf(" K |");
                break;
            case BlackKing:
                printf(" k |");
                break;
            default:
                printf("   |");
                break;
            }
        }
        printf("\n");
        printf("   +---+---+---+---+---+---+---+---+");
        printf("\n");
    }

    printf("    ");
    for (int j = 0; j < CHESS_BOARD_COL_CNT; j++)
    {
        printf(" %c  ", get_col_fr_adj_col(j));
    }

    printf("\n\n");
}

void one_hot_encode_board(int *board, int *out)
{
    memset(out, 0, sizeof(int) * CHESS_ONE_HOT_ENCODED_BOARD_LEN);

    for (int i = 0; i < CHESS_BOARD_LEN; i++)
    {
        switch (board[i])
        {
        case WhitePawn:
            out[i * CHESS_ONE_HOT_ENCODE_COMBINATION_CNT + 0] = 1;
            break;
        case WhiteKnight:
            out[i * CHESS_ONE_HOT_ENCODE_COMBINATION_CNT + 1] = 1;
            break;
        case WhiteBishop:
            out[i * CHESS_ONE_HOT_ENCODE_COMBINATION_CNT + 2] = 1;
            break;
        case WhiteRook:
            out[i * CHESS_ONE_HOT_ENCODE_COMBINATION_CNT + 3] = 1;
            break;
        case WhiteQueen:
            out[i * CHESS_ONE_HOT_ENCODE_COMBINATION_CNT + 4] = 1;
            break;
        case WhiteKing:
            out[i * CHESS_ONE_HOT_ENCODE_COMBINATION_CNT + 5] = 1;
            break;
        case BlackPawn:
            out[i * CHESS_ONE_HOT_ENCODE_COMBINATION_CNT + 6] = 1;
            break;
        case BlackKnight:
            out[i * CHESS_ONE_HOT_ENCODE_COMBINATION_CNT + 7] = 1;
            break;
        case BlackBishop:
            out[i * CHESS_ONE_HOT_ENCODE_COMBINATION_CNT + 8] = 1;
            break;
        case BlackRook:
            out[i * CHESS_ONE_HOT_ENCODE_COMBINATION_CNT + 9] = 1;
            break;
        case BlackQueen:
            out[i * CHESS_ONE_HOT_ENCODE_COMBINATION_CNT + 10] = 1;
            break;
        case BlackKing:
            out[i * CHESS_ONE_HOT_ENCODE_COMBINATION_CNT + 11] = 1;
            break;
        default: // Empty space.
            out[i * CHESS_ONE_HOT_ENCODE_COMBINATION_CNT + 12] = 1;
            break;
        }
    }
}

int eval_board(int *board)
{
    int sum = 0;

    for (int i = 0; i < CHESS_BOARD_LEN; i++)
    {
        sum += board[i];
    }

    return sum;
}

int get_worst_case(int *board, bool white_flg, bool cur_white_flg, int depth, int cur_depth)
{
    if (is_in_checkmate(board, !white_flg))
    {
        if (white_flg)
        {
            return 1000;
        }
        else
        {
            return -1000;
        }
    }

    if (cur_depth == depth)
    {
        return eval_board(board);
    }

    int legal_moves[CHESS_MAX_LEGAL_MOVE_CNT];

    int sim_board[CHESS_BOARD_LEN];

    int worst_eval;

    if (white_flg)
    {
        worst_eval = INT_MAX;
    }
    else
    {
        worst_eval = -INT_MAX;
    }

    if (cur_white_flg)
    {
        for (int piece_idx = 0; piece_idx < CHESS_BOARD_LEN; piece_idx++)
        {
            if (is_piece_white((ChessPiece)board[piece_idx]))
            {
                get_legal_moves(board, piece_idx, legal_moves, true);

                for (int mov_idx = 0; mov_idx < CHESS_MAX_LEGAL_MOVE_CNT; mov_idx++)
                {
                    if (legal_moves[mov_idx] == CHESS_INVALID_VALUE)
                    {
                        break;
                    }

                    simulate_board_change_w_srcdst_idx(board, piece_idx, legal_moves[mov_idx], sim_board);

                    int eval = get_worst_case(sim_board, white_flg, !cur_white_flg, depth, cur_depth + 1);

                    if (white_flg)
                    {
                        if (eval < worst_eval)
                        {
                            worst_eval = eval;
                        }
                    }
                    else
                    {
                        if (eval > worst_eval)
                        {
                            worst_eval = eval;
                        }
                    }
                }
            }
        }
    }
    else
    {
        for (int piece_idx = 0; piece_idx < CHESS_BOARD_LEN; piece_idx++)
        {
            if (is_piece_black((ChessPiece)board[piece_idx]) == 1)
            {
                get_legal_moves(board, piece_idx, legal_moves, true);

                for (int mov_idx = 0; mov_idx < CHESS_MAX_LEGAL_MOVE_CNT; mov_idx++)
                {
                    if (legal_moves[mov_idx] == CHESS_INVALID_VALUE)
                    {
                        break;
                    }

                    simulate_board_change_w_srcdst_idx(board, piece_idx, legal_moves[mov_idx], sim_board);

                    int eval = get_worst_case(sim_board, white_flg, !cur_white_flg, depth, cur_depth + 1);

                    if (white_flg)
                    {
                        if (eval < worst_eval)
                        {
                            worst_eval = eval;
                        }
                    }
                    else
                    {
                        if (eval > worst_eval)
                        {
                            worst_eval = eval;
                        }
                    }
                }
            }
        }
    }

    // Make sure there were moves to evaluate.
    if (worst_eval == INT_MAX || worst_eval == -INT_MAX)
    {
        worst_eval = 0;
    }

    return eval_board(board) + worst_eval;
}

Tensor *process_convolutions(int *board)
{
    int moves[CHESS_MAX_LEGAL_MOVE_CNT];

    Tensor *out = new Tensor(8, 8, Cpu, board);

    int *cpy_board = copy_board(board, NULL);
    int piece_cnt = 0;

    for (int piece_idx = 0; piece_idx < CHESS_BOARD_LEN; piece_idx++)
    {
        ChessPiece piece = (ChessPiece)board[piece_idx];

        if (piece != Empty)
        {
            get_piece_influence(board, piece_idx, moves);

            for (int mov_idx = 0; mov_idx < CHESS_MAX_LEGAL_MOVE_CNT; mov_idx++)
            {
                int mov_dst_idx = moves[mov_idx];

                if (mov_dst_idx == CHESS_INVALID_VALUE)
                {
                    break;
                }

                float val = out->get_val(mov_dst_idx);

                if (is_piece_white(piece))
                {
                    val += ((abs((int)board[mov_dst_idx]) * 1.0f) + 0.1f);
                }
                else if (is_piece_black(piece))
                {
                    val += (-(abs((int)board[mov_dst_idx]) * 1.0f) + -0.1f);
                }

                out->set_val(mov_dst_idx, val);
            }
        }
    }

    return out;
}