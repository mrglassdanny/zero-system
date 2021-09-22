#include "pgn.cuh"

void PGNImport_add(PGNImport *pgn, PGNMoveList *list)
{
    if (pgn->cnt == pgn->cap)
    {
        PGNMoveList **nxt = (PGNMoveList **)malloc(sizeof(PGNMoveList *) * (pgn->cap + pgn->alc));
        memcpy(nxt, pgn->games, sizeof(PGNMoveList *) * (pgn->cap));
        free(pgn->games);
        pgn->games = nxt;
        pgn->cap += pgn->alc;
    }

    pgn->games[pgn->cnt++] = list;
}

PGNMoveList *PGNMoveList_init()
{
    PGNMoveList *list = (PGNMoveList *)malloc(sizeof(PGNMoveList));

    list->white_won_flg = 0;
    list->black_won_flg = 0;

    memset(list->arr, 0, (CHESS_MAX_GAME_MOVE_CNT * CHESS_MAX_MOVE_LEN));
    list->cnt = 0;

    return list;
}

void PGNMoveList_add(PGNMoveList *list, const char *mov)
{
    memcpy(list->arr[list->cnt++], mov, CHESS_MAX_MOVE_LEN);
}

void PGNMoveList_reset(PGNMoveList *list)
{
    list->white_won_flg = 0;
    list->black_won_flg = 0;
    memset(list->arr, 0, (CHESS_MAX_GAME_MOVE_CNT * CHESS_MAX_MOVE_LEN));
    list->cnt = 0;
}

PGNImport *PGNImport_init(const char *pgn_file_name)
{

    PGNImport *pgn = (PGNImport *)malloc(sizeof(PGNImport));
    pgn->cnt = 0;
    pgn->alc = 1000;
    pgn->cap = pgn->alc;
    pgn->games = (PGNMoveList **)malloc(sizeof(PGNMoveList *) * pgn->cap);

    FILE *file_ptr = fopen(pgn_file_name, "rb");

    fseek(file_ptr, 0L, SEEK_END);
    long file_size = ftell(file_ptr);
    rewind(file_ptr);

    char *buf = (char *)malloc(file_size);
    fread(buf, 1, file_size, file_ptr);

    fclose(file_ptr);

    PGNMoveList *list = PGNMoveList_init();

    char mov[CHESS_MAX_MOVE_LEN] = {0};

    for (int i = 0; i < file_size; i++)
    {
        if (i - 2 > 0 && buf[i - 2] == '\n' && buf[i - 1] == '1' && buf[i] == '.')
        {
            i++;
            // At start of game (past "1.")).

            // Read moves.

            while ((i + 1) < file_size && buf[i] != ' ' && buf[i + 1] != ' ')
            {

                // Turn x.

                // White move.
                int move_ctr = 0;
                while (i < file_size && buf[i] != ' ' && buf[i] != '\n' && buf[i] != '\r')
                {
                    mov[move_ctr++] = buf[i++];
                }
                PGNMoveList_add(list, mov);
                memset(mov, 0, CHESS_MAX_MOVE_LEN);
                i++;

                // It is possible that white made the last move.
                if ((i + 1) < file_size && buf[i] != ' ' && buf[i + 1] != ' ')
                {

                    //Black move.
                    move_ctr = 0;
                    while (i < file_size && buf[i] != ' ' && buf[i] != '\n' && buf[i] != '\r')
                    {
                        mov[move_ctr++] = buf[i++];
                    }
                    PGNMoveList_add(list, mov);
                    memset(mov, 0, CHESS_MAX_MOVE_LEN);

                    // Go to next turn.
                    if ((i + 1) < file_size && buf[i + 1] != ' ')
                    {
                        while (i < file_size && buf[i] != '.')
                        {
                            i++;
                        }
                        i++;
                    }
                }
            }

            // At end of game (right before 1-0 or 0-1 or 1/2-1/2).
            // Should be spaces.
            i++;
            i++;
            if (buf[i] == '0')
            {
                // White loss.
                list->white_won_flg = 0;
                list->black_won_flg = 1;
            }
            else
            {
                // buf[i] == 1;
                // This could mean tie; let's check next char for '/'.
                if (buf[i + 1] == '/')
                {
                    // Tie -- do nothing!
                }
                else
                {
                    // White win.
                    list->white_won_flg = 1;
                    list->black_won_flg = 0;
                }
            }

            PGNImport_add(pgn, list);
            list = PGNMoveList_init();
        }

        // Ignore characters here...
    }

    free(buf);

    return pgn;
}

void PGNImport_free(PGNImport *pgn)
{
    for (int i = 0; i < pgn->cnt; i++)
    {
        free(pgn->games[i]);
    }
    free(pgn->games);
    free(pgn);
}