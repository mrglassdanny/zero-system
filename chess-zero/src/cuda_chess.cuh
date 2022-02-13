#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#include <vector>

#include <zero_system/mod.cuh>

#include "chess.cuh"

void cuda_play(Model *model);