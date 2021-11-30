#include "cuda_chess.cuh"

#define CHESS_BOARD_ROW_CNT 8
#define CHESS_BOARD_COL_CNT 8
#define CHESS_BOARD_LEN (CHESS_BOARD_COL_CNT * CHESS_BOARD_ROW_CNT)

#define CHESS_MAX_LEGAL_MOVE_CNT 64

#define CHESS_MAX_MOVE_LEN 8
#define CHESS_MAX_GAME_MOVE_CNT 500

#define CHESS_INVALID_VALUE -1

#define CHESS_ONE_HOT_ENCODE_COMBINATION_CNT 6
#define CHESS_ONE_HOT_ENCODED_BOARD_LEN (CHESS_BOARD_LEN * CHESS_ONE_HOT_ENCODE_COMBINATION_CNT)

typedef enum ChessPiece
{
    Empty = 0,
    WhitePawn = 1,
    WhiteKnight = 3,
    WhiteBishop = 4,
    WhiteRook = 6,
    WhiteQueen = 9,
    WhiteKing = 10,
    BlackPawn = -1,
    BlackKnight = -3,
    BlackBishop = -4,
    BlackRook = -6,
    BlackQueen = -9,
    BlackKing = -10
} ChessPiece;

// Device functions:

__device__ bool d_is_piece_white(ChessPiece piece)
{
    return piece > 0;
}

__device__ bool d_is_piece_black(ChessPiece piece)
{
    return piece < 0;
}

__device__ bool d_is_piece_same_color(ChessPiece a, ChessPiece b)
{
    if ((d_is_piece_white(a) && d_is_piece_white(b)) || (d_is_piece_black(a) && d_is_piece_black(b)))
    {
        return true;
    }
    else
    {
        return false;
    }
}

__device__ int d_get_adj_col_fr_idx(int idx)
{
    return idx % CHESS_BOARD_COL_CNT;
}

__device__ int d_get_adj_row_fr_idx(int idx)
{
    return idx / CHESS_BOARD_ROW_CNT;
}

__device__ int d_get_idx_fr_adj_col_adj_row(int adj_col, int adj_row)
{
    return (adj_row * CHESS_BOARD_COL_CNT) + adj_col;
}

__device__ bool d_is_adj_row_valid(int adj_row)
{
    if (adj_row >= 0 && adj_row < CHESS_BOARD_ROW_CNT)
    {
        return true;
    }
    else
    {
        return false;
    }
}

__device__ bool d_is_adj_col_adj_row_valid(int adj_col, int adj_row)
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

__device__ void d_get_legal_moves(int *board, int piece_idx, int *out, bool test_in_check_flg)
{
    memset(out, CHESS_INVALID_VALUE, sizeof(int) * CHESS_MAX_LEGAL_MOVE_CNT);
    int mov_ctr = 0;

    ChessPiece piece = (ChessPiece)board[piece_idx];

    int adj_col = d_get_adj_col_fr_idx(piece_idx);
    int adj_row = d_get_adj_row_fr_idx(piece_idx);

    int white_mov_flg;
    if (d_is_piece_white(piece))
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
    case ChessPiece::WhitePawn:
        // TODO: au passant
        {
            test_idx = d_get_idx_fr_adj_col_adj_row(adj_col, adj_row + 1);
            if (d_is_adj_row_valid(adj_row + 1) && board[test_idx] == ChessPiece::Empty)
            {
                out[mov_ctr++] = test_idx;
            }

            test_idx = d_get_idx_fr_adj_col_adj_row(adj_col - 1, adj_row + 1);
            if (d_is_adj_col_adj_row_valid(adj_col - 1, adj_row + 1) && board[test_idx] != ChessPiece::Empty && !d_is_piece_same_color(piece, (ChessPiece)board[test_idx]))
            {
                out[mov_ctr++] = test_idx;
            }

            test_idx = d_get_idx_fr_adj_col_adj_row(adj_col + 1, adj_row + 1);
            if (d_is_adj_col_adj_row_valid(adj_col + 1, adj_row + 1) && board[test_idx] != ChessPiece::Empty && !d_is_piece_same_color(piece, (ChessPiece)board[test_idx]))
            {
                out[mov_ctr++] = test_idx;
            }

            if (adj_row == 1)
            {
                // Dont need to check if row adjustments are valid since we know that starting row is 2.
                test_idx = d_get_idx_fr_adj_col_adj_row(adj_col, adj_row + 1);
                if (board[test_idx] == ChessPiece::Empty)
                {
                    test_idx = d_get_idx_fr_adj_col_adj_row(adj_col, adj_row + 2);
                    if (board[test_idx] == ChessPiece::Empty)
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }
            }
        }

        break;
    case ChessPiece::BlackPawn:
        // TODO: au passant
        {
            test_idx = d_get_idx_fr_adj_col_adj_row(adj_col, adj_row - 1);
            if (d_is_adj_row_valid(adj_row - 1) && board[test_idx] == ChessPiece::Empty)
            {
                out[mov_ctr++] = test_idx;
            }

            test_idx = d_get_idx_fr_adj_col_adj_row(adj_col - 1, adj_row - 1);
            if (d_is_adj_col_adj_row_valid(adj_col - 1, adj_row - 1) && board[test_idx] != ChessPiece::Empty && !d_is_piece_same_color(piece, (ChessPiece)board[test_idx]))
            {
                out[mov_ctr++] = test_idx;
            }

            test_idx = d_get_idx_fr_adj_col_adj_row(adj_col + 1, adj_row - 1);
            if (d_is_adj_col_adj_row_valid(adj_col + 1, adj_row - 1) && board[test_idx] != ChessPiece::Empty && !d_is_piece_same_color(piece, (ChessPiece)board[test_idx]))
            {
                out[mov_ctr++] = test_idx;
            }

            if (adj_row == 6)
            {
                // Dont need to check if row adjustments are valid since we know that starting row is 7.
                test_idx = d_get_idx_fr_adj_col_adj_row(adj_col, adj_row - 1);
                if (board[test_idx] == ChessPiece::Empty)
                {
                    test_idx = d_get_idx_fr_adj_col_adj_row(adj_col, adj_row - 2);
                    if (board[test_idx] == ChessPiece::Empty)
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }
            }
        }

        break;
    case ChessPiece::WhiteKnight:
    case ChessPiece::BlackKnight:
    {

        if (d_is_adj_col_adj_row_valid(adj_col + 1, adj_row + 2))
        {
            test_idx = d_get_idx_fr_adj_col_adj_row(adj_col + 1, adj_row + 2);
            if (!d_is_piece_same_color(piece, (ChessPiece)board[test_idx]))
            {
                out[mov_ctr++] = test_idx;
            }
        }

        if (d_is_adj_col_adj_row_valid(adj_col + 1, adj_row - 2))
        {
            test_idx = d_get_idx_fr_adj_col_adj_row(adj_col + 1, adj_row - 2);
            if (!d_is_piece_same_color(piece, (ChessPiece)board[test_idx]))
            {
                out[mov_ctr++] = test_idx;
            }
        }

        if (d_is_adj_col_adj_row_valid(adj_col - 1, adj_row + 2))
        {
            test_idx = d_get_idx_fr_adj_col_adj_row(adj_col - 1, adj_row + 2);
            if (!d_is_piece_same_color(piece, (ChessPiece)board[test_idx]))
            {
                out[mov_ctr++] = test_idx;
            }
        }

        if (d_is_adj_col_adj_row_valid(adj_col - 1, adj_row - 2))
        {
            test_idx = d_get_idx_fr_adj_col_adj_row(adj_col - 1, adj_row - 2);
            if (!d_is_piece_same_color(piece, (ChessPiece)board[test_idx]))
            {
                out[mov_ctr++] = test_idx;
            }
        }

        if (d_is_adj_col_adj_row_valid(adj_col + 2, adj_row + 1))
        {
            test_idx = d_get_idx_fr_adj_col_adj_row(adj_col + 2, adj_row + 1);
            if (!d_is_piece_same_color(piece, (ChessPiece)board[test_idx]))
            {
                out[mov_ctr++] = test_idx;
            }
        }

        if (d_is_adj_col_adj_row_valid(adj_col + 2, adj_row - 1))
        {
            test_idx = d_get_idx_fr_adj_col_adj_row(adj_col + 2, adj_row - 1);
            if (!d_is_piece_same_color(piece, (ChessPiece)board[test_idx]))
            {
                out[mov_ctr++] = test_idx;
            }
        }

        if (d_is_adj_col_adj_row_valid(adj_col - 2, adj_row + 1))
        {
            test_idx = d_get_idx_fr_adj_col_adj_row(adj_col - 2, adj_row + 1);
            if (!d_is_piece_same_color(piece, (ChessPiece)board[test_idx]))
            {
                out[mov_ctr++] = test_idx;
            }
        }

        if (d_is_adj_col_adj_row_valid(adj_col - 2, adj_row - 1))
        {
            test_idx = d_get_idx_fr_adj_col_adj_row(adj_col - 2, adj_row - 1);
            if (!d_is_piece_same_color(piece, (ChessPiece)board[test_idx]))
            {
                out[mov_ctr++] = test_idx;
            }
        }
    }

    break;
    case ChessPiece::WhiteBishop:
    case ChessPiece::BlackBishop:
    {
        int ne = 0;
        int sw = 0;
        int se = 0;
        int nw = 0;
        for (int i = 1; i < 8; i++)
        {

            if (d_is_adj_col_adj_row_valid(adj_col + i, adj_row + i) && ne == 0)
            {
                test_idx = d_get_idx_fr_adj_col_adj_row(adj_col + i, adj_row + i);
                if (board[test_idx] != ChessPiece::Empty)
                {
                    ne = 1;
                    if (!d_is_piece_same_color(piece, (ChessPiece)board[test_idx]))
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }
                else
                {
                    out[mov_ctr++] = test_idx;
                }
            }

            if (d_is_adj_col_adj_row_valid(adj_col - i, adj_row - i) && sw == 0)
            {
                test_idx = d_get_idx_fr_adj_col_adj_row(adj_col - i, adj_row - i);
                if (board[test_idx] != ChessPiece::Empty)
                {
                    sw = 1;
                    if (!d_is_piece_same_color(piece, (ChessPiece)board[test_idx]))
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }
                else
                {
                    out[mov_ctr++] = test_idx;
                }
            }

            if (d_is_adj_col_adj_row_valid(adj_col + i, adj_row - i) && se == 0)
            {
                test_idx = d_get_idx_fr_adj_col_adj_row(adj_col + i, adj_row - i);
                if (board[test_idx] != ChessPiece::Empty)
                {
                    se = 1;
                    if (!d_is_piece_same_color(piece, (ChessPiece)board[test_idx]))
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }
                else
                {
                    out[mov_ctr++] = test_idx;
                }
            }

            if (d_is_adj_col_adj_row_valid(adj_col - i, adj_row + i) && nw == 0)
            {
                test_idx = d_get_idx_fr_adj_col_adj_row(adj_col - i, adj_row + i);
                if (board[test_idx] != ChessPiece::Empty)
                {
                    nw = 1;
                    if (!d_is_piece_same_color(piece, (ChessPiece)board[test_idx]))
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
    case ChessPiece::WhiteRook:
    case ChessPiece::BlackRook:
    {
        int n = 0;
        int s = 0;
        int e = 0;
        int w = 0;
        for (int i = 1; i < 8; i++)
        {

            if (d_is_adj_col_adj_row_valid(adj_col + i, adj_row) && e == 0)
            {
                test_idx = d_get_idx_fr_adj_col_adj_row(adj_col + i, adj_row);
                if (board[test_idx] != ChessPiece::Empty)
                {
                    e = 1;
                    if (!d_is_piece_same_color(piece, (ChessPiece)board[test_idx]))
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }
                else
                {
                    out[mov_ctr++] = test_idx;
                }
            }

            if (d_is_adj_col_adj_row_valid(adj_col - i, adj_row) && w == 0)
            {
                test_idx = d_get_idx_fr_adj_col_adj_row(adj_col - i, adj_row);
                if (board[test_idx] != ChessPiece::Empty)
                {
                    w = 1;
                    if (!d_is_piece_same_color(piece, (ChessPiece)board[test_idx]))
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }
                else
                {
                    out[mov_ctr++] = test_idx;
                }
            }

            if (d_is_adj_col_adj_row_valid(adj_col, adj_row + i) && n == 0)
            {
                test_idx = d_get_idx_fr_adj_col_adj_row(adj_col, adj_row + i);
                if (board[test_idx] != ChessPiece::Empty)
                {
                    n = 1;
                    if (!d_is_piece_same_color(piece, (ChessPiece)board[test_idx]))
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }
                else
                {
                    out[mov_ctr++] = test_idx;
                }
            }

            if (d_is_adj_col_adj_row_valid(adj_col, adj_row - i) && s == 0)
            {
                test_idx = d_get_idx_fr_adj_col_adj_row(adj_col, adj_row - i);
                if (board[test_idx] != ChessPiece::Empty)
                {
                    s = 1;
                    if (!d_is_piece_same_color(piece, (ChessPiece)board[test_idx]))
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
    case ChessPiece::WhiteQueen:
    case ChessPiece::BlackQueen:
        // ne,sw,se,nw
        {
            int ne = 0;
            int sw = 0;
            int se = 0;
            int nw = 0;
            for (int i = 1; i < 8; i++)
            {

                if (d_is_adj_col_adj_row_valid(adj_col + i, adj_row + i) && ne == 0)
                {
                    test_idx = d_get_idx_fr_adj_col_adj_row(adj_col + i, adj_row + i);
                    if (board[test_idx] != ChessPiece::Empty)
                    {
                        ne = 1;
                        if (!d_is_piece_same_color(piece, (ChessPiece)board[test_idx]))
                        {
                            out[mov_ctr++] = test_idx;
                        }
                    }
                    else
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }

                if (d_is_adj_col_adj_row_valid(adj_col - i, adj_row - i) && sw == 0)
                {
                    test_idx = d_get_idx_fr_adj_col_adj_row(adj_col - i, adj_row - i);
                    if (board[test_idx] != ChessPiece::Empty)
                    {
                        sw = 1;
                        if (!d_is_piece_same_color(piece, (ChessPiece)board[test_idx]))
                        {
                            out[mov_ctr++] = test_idx;
                        }
                    }
                    else
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }

                if (d_is_adj_col_adj_row_valid(adj_col + i, adj_row - i) && se == 0)
                {
                    test_idx = d_get_idx_fr_adj_col_adj_row(adj_col + i, adj_row - i);
                    if (board[test_idx] != ChessPiece::Empty)
                    {
                        se = 1;
                        if (!d_is_piece_same_color(piece, (ChessPiece)board[test_idx]))
                        {
                            out[mov_ctr++] = test_idx;
                        }
                    }
                    else
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }

                if (d_is_adj_col_adj_row_valid(adj_col - i, adj_row + i) && nw == 0)
                {
                    test_idx = d_get_idx_fr_adj_col_adj_row(adj_col - i, adj_row + i);
                    if (board[test_idx] != ChessPiece::Empty)
                    {
                        nw = 1;
                        if (!d_is_piece_same_color(piece, (ChessPiece)board[test_idx]))
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

                if (d_is_adj_col_adj_row_valid(adj_col + i, adj_row) && e == 0)
                {
                    test_idx = d_get_idx_fr_adj_col_adj_row(adj_col + i, adj_row);
                    if (board[test_idx] != ChessPiece::Empty)
                    {
                        e = 1;
                        if (!d_is_piece_same_color(piece, (ChessPiece)board[test_idx]))
                        {
                            out[mov_ctr++] = test_idx;
                        }
                    }
                    else
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }

                if (d_is_adj_col_adj_row_valid(adj_col - i, adj_row) && w == 0)
                {
                    test_idx = d_get_idx_fr_adj_col_adj_row(adj_col - i, adj_row);
                    if (board[test_idx] != ChessPiece::Empty)
                    {
                        w = 1;
                        if (!d_is_piece_same_color(piece, (ChessPiece)board[test_idx]))
                        {
                            out[mov_ctr++] = test_idx;
                        }
                    }
                    else
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }

                if (d_is_adj_col_adj_row_valid(adj_col, adj_row + i) && n == 0)
                {
                    test_idx = d_get_idx_fr_adj_col_adj_row(adj_col, adj_row + i);
                    if (board[test_idx] != ChessPiece::Empty)
                    {
                        n = 1;
                        if (!d_is_piece_same_color(piece, (ChessPiece)board[test_idx]))
                        {
                            out[mov_ctr++] = test_idx;
                        }
                    }
                    else
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }

                if (d_is_adj_col_adj_row_valid(adj_col, adj_row - i) && s == 0)
                {
                    test_idx = d_get_idx_fr_adj_col_adj_row(adj_col, adj_row - i);
                    if (board[test_idx] != ChessPiece::Empty)
                    {
                        s = 1;
                        if (!d_is_piece_same_color(piece, (ChessPiece)board[test_idx]))
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
    case ChessPiece::WhiteKing:
    case ChessPiece::BlackKing:
        // ne,sw,se,nw
        {
            int ne = 0;
            int sw = 0;
            int se = 0;
            int nw = 0;
            for (int i = 1; i < 2; i++)
            {

                if (d_is_adj_col_adj_row_valid(adj_col + i, adj_row + i) && ne == 0)
                {
                    test_idx = d_get_idx_fr_adj_col_adj_row(adj_col + i, adj_row + i);
                    if (board[test_idx] != ChessPiece::Empty)
                    {
                        ne = 1;
                        if (!d_is_piece_same_color(piece, (ChessPiece)board[test_idx]))
                        {
                            out[mov_ctr++] = test_idx;
                        }
                    }
                    else
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }

                if (d_is_adj_col_adj_row_valid(adj_col - i, adj_row - i) && sw == 0)
                {
                    test_idx = d_get_idx_fr_adj_col_adj_row(adj_col - i, adj_row - i);
                    if (board[test_idx] != ChessPiece::Empty)
                    {
                        sw = 1;
                        if (!d_is_piece_same_color(piece, (ChessPiece)board[test_idx]))
                        {
                            out[mov_ctr++] = test_idx;
                        }
                    }
                    else
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }

                if (d_is_adj_col_adj_row_valid(adj_col + i, adj_row - i) && se == 0)
                {
                    test_idx = d_get_idx_fr_adj_col_adj_row(adj_col + i, adj_row - i);
                    if (board[test_idx] != ChessPiece::Empty)
                    {
                        se = 1;
                        if (!d_is_piece_same_color(piece, (ChessPiece)board[test_idx]))
                        {
                            out[mov_ctr++] = test_idx;
                        }
                    }
                    else
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }

                if (d_is_adj_col_adj_row_valid(adj_col - i, adj_row + i) && nw == 0)
                {
                    test_idx = d_get_idx_fr_adj_col_adj_row(adj_col - i, adj_row + i);
                    if (board[test_idx] != ChessPiece::Empty)
                    {
                        nw = 1;
                        if (!d_is_piece_same_color(piece, (ChessPiece)board[test_idx]))
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

                if (d_is_adj_col_adj_row_valid(adj_col + i, adj_row) && e == 0)
                {
                    test_idx = d_get_idx_fr_adj_col_adj_row(adj_col + i, adj_row);
                    if (board[test_idx] != ChessPiece::Empty)
                    {
                        e = 1;
                        if (!d_is_piece_same_color(piece, (ChessPiece)board[test_idx]))
                        {
                            out[mov_ctr++] = test_idx;
                        }
                    }
                    else
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }

                if (d_is_adj_col_adj_row_valid(adj_col - i, adj_row) && w == 0)
                {
                    test_idx = d_get_idx_fr_adj_col_adj_row(adj_col - i, adj_row);
                    if (board[test_idx] != ChessPiece::Empty)
                    {
                        w = 1;
                        if (!d_is_piece_same_color(piece, (ChessPiece)board[test_idx]))
                        {
                            out[mov_ctr++] = test_idx;
                        }
                    }
                    else
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }

                if (d_is_adj_col_adj_row_valid(adj_col, adj_row + i) && n == 0)
                {
                    test_idx = d_get_idx_fr_adj_col_adj_row(adj_col, adj_row + i);
                    if (board[test_idx] != ChessPiece::Empty)
                    {
                        n = 1;
                        if (!d_is_piece_same_color(piece, (ChessPiece)board[test_idx]))
                        {
                            out[mov_ctr++] = test_idx;
                        }
                    }
                    else
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }

                if (d_is_adj_col_adj_row_valid(adj_col, adj_row - i) && s == 0)
                {
                    test_idx = d_get_idx_fr_adj_col_adj_row(adj_col, adj_row - i);
                    if (board[test_idx] != ChessPiece::Empty)
                    {
                        s = 1;
                        if (!d_is_piece_same_color(piece, (ChessPiece)board[test_idx]))
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
        if (piece == ChessPiece::WhiteKing)
        {
            if (adj_col == 4 && adj_row == 0)
            {
                // Queen side castle.
                if (board[0] == ChessPiece::WhiteRook)
                {
                    if (board[1] == ChessPiece::Empty && board[2] == ChessPiece::Empty && board[3] == ChessPiece::Empty &&
                        !d_is_cell_under_attack(board, 1, true) && !d_is_cell_under_attack(board, 2, true) && !d_is_cell_under_attack(board, 3, true))
                    {
                        out[mov_ctr++] = 2;
                    }
                }

                // King side castle.
                if (board[7] == ChessPiece::WhiteRook)
                {
                    if (board[6] == ChessPiece::Empty && board[5] == ChessPiece::Empty &&
                        !d_is_cell_under_attack(board, 6, true) && !d_is_cell_under_attack(board, 5, true))
                    {
                        out[mov_ctr++] = 5;
                    }
                }
            }
        }
        else
        {
            if (adj_col == 4 && adj_row == 7)
            {
                // Queen side castle.
                if (board[56] == ChessPiece::BlackRook)
                {
                    if (board[57] == ChessPiece::Empty && board[58] == ChessPiece::Empty && board[59] == ChessPiece::Empty &&
                        !d_is_cell_under_attack(board, 57, false) && !d_is_cell_under_attack(board, 58, false) && !d_is_cell_under_attack(board, 59, false))
                    {
                        out[mov_ctr++] = 58;
                    }
                }

                // King side castle.
                if (board[63] == ChessPiece::BlackRook)
                {
                    if (board[62] == ChessPiece::Empty && board[61] == ChessPiece::Empty &&
                        !d_is_cell_under_attack(board, 62, false) && !d_is_cell_under_attack(board, 61, false))
                    {
                        out[mov_ctr++] = 61;
                    }
                }
            }
        }

        break;
    default: // Nothing...
        break;
    }
}

// Change board with move (src idx & dst idx)

// Kernel functions:

__global__ void k_reset_board(float *board)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < CHESS_BOARD_LEN)
    {
        switch (tid)
        {
        case 0:
        case 7:
            board[tid] = ChessPiece::WhiteRook;
            break;
        case 1:
        case 6:
            board[tid] = ChessPiece::WhiteKnight;
            break;
        case 2:
        case 5:
            board[tid] = ChessPiece::WhiteBishop;
            break;
        case 3:
            board[tid] = ChessPiece::WhiteQueen;
            break;
        case 4:
            board[tid] = ChessPiece::WhiteKing;
            break;
        case 8:
        case 9:
        case 10:
        case 11:
        case 12:
        case 13:
        case 14:
        case 15:
            board[tid] = ChessPiece::WhitePawn;
            break;
        case 56:
        case 63:
            board[tid] = ChessPiece::BlackRook;
            break;
        case 57:
        case 62:
            board[tid] = ChessPiece::BlackKnight;
            break;
        case 58:
        case 61:
            board[tid] = ChessPiece::BlackBishop;
            break;
        case 59:
            board[tid] = ChessPiece::BlackQueen;
            break;
        case 60:
            board[tid] = ChessPiece::BlackKing;
            break;
        case 48:
        case 49:
        case 50:
        case 51:
        case 52:
        case 53:
        case 54:
        case 55:
            board[tid] = ChessPiece::BlackPawn;
            break;
        default:
            board[tid] = ChessPiece::Empty;
            break;
        }
    }
}

__global__ void k_clear_one_hot_board(float *board)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < CHESS_ONE_HOT_ENCODED_BOARD_LEN)
    {
        board[tid] = 0.0f;
    }
}

// One hot encode
__global__ void k_one_hot_encode_board(int *board, float *out)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < CHESS_BOARD_LEN)
    {
        switch ((ChessPiece)board[tid])
        {
        case ChessPiece::WhitePawn:
            out[tid] = 1.0f;
            break;
        case ChessPiece::WhiteKnight:
            out[tid + CHESS_BOARD_LEN] = 1.0f;
            break;
        case ChessPiece::WhiteBishop:
            out[tid + (CHESS_BOARD_LEN * 2)] = 1.0f;
            break;
        case ChessPiece::WhiteRook:
            out[tid + (CHESS_BOARD_LEN * 3)] = 1.0f;
            break;
        case ChessPiece::WhiteQueen:
            out[tid + (CHESS_BOARD_LEN * 4)] = 1.0f;
            break;
        case ChessPiece::WhiteKing:
            out[tid + (CHESS_BOARD_LEN * 5)] = 1.0f;
            break;
        case ChessPiece::BlackPawn:
            out[tid] = -1.0f;
            break;
        case ChessPiece::BlackKnight:
            out[tid + (CHESS_BOARD_LEN)] = -1.0f;
            break;
        case ChessPiece::BlackBishop:
            out[tid + (CHESS_BOARD_LEN * 2)] = -1.0f;
            break;
        case ChessPiece::BlackRook:
            out[tid + (CHESS_BOARD_LEN * 3)] = -1.0f;
            break;
        case ChessPiece::BlackQueen:
            out[tid + (CHESS_BOARD_LEN * 4)] = -1.0f;
            break;
        case ChessPiece::BlackKing:
            out[tid + (CHESS_BOARD_LEN * 5)] = -1.0f;
            break;
        default: // ChessPiece::Empty:
            break;
        }
    }
}

// Check
__global__ void k_is_in_check(int *board, bool white_mov_flg, bool *flg)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < CHESS_BOARD_LEN)
    {
        int legal_moves[CHESS_MAX_LEGAL_MOVE_CNT];

        if (white_mov_flg)
        {
            if (is_piece_black((ChessPiece)board[tid]))
            {
                d_get_legal_moves(board, tid, legal_moves, false);

                for (int mov_idx = 0; mov_idx < CHESS_MAX_LEGAL_MOVE_CNT; mov_idx++)
                {
                    if (legal_moves[mov_idx] == CHESS_INVALID_VALUE)
                    {
                        return;
                    }

                    if ((ChessPiece)board[legal_moves[mov_idx]] == ChessPiece::WhiteKing)
                    {
                        *flg = true;
                        return;
                    }
                }
            }
        }
        else
        {
            if (is_piece_white((ChessPiece)board[tid]))
            {
                d_get_legal_moves(board, tid, legal_moves, false);

                for (int mov_idx = 0; mov_idx < CHESS_MAX_LEGAL_MOVE_CNT; mov_idx++)
                {
                    if (legal_moves[mov_idx] == CHESS_INVALID_VALUE)
                    {
                        return;
                    }

                    if ((ChessPiece)board[legal_moves[mov_idx]] == ChessPiece::BlackKing)
                    {
                        *flg = true;
                        return;
                    }
                }
            }
        }
    }
}

// Checkmate/Stalemate
__global__ void k_has_no_legal_moves(int *board, bool white_mov_flg, bool *flg)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < CHESS_BOARD_LEN)
    {
        int legal_moves[CHESS_MAX_LEGAL_MOVE_CNT];

        if (white_mov_flg)
        {
            if (d_is_piece_white((ChessPiece)board[tid]))
            {
                d_get_legal_moves(board, tid, legal_moves, true);

                if (legal_moves[0] != CHESS_INVALID_VALUE)
                {
                    *flg = false;
                    return;
                }
            }
        }
        else
        {
            if (d_is_piece_black((ChessPiece)board[tid]))
            {
                d_get_legal_moves(board, tid, legal_moves, true);

                if (legal_moves[0] != CHESS_INVALID_VALUE)
                {
                    *flg = false;
                    return;
                }
            }
        }
    }
}

// Rotate

// Reflect