#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <conio.h>
#include <random>
#include <vector>
#include <map>
#include <windows.h>

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include <device_atomic_functions.h>

namespace zero
{
    namespace core
    {
        class FileUtils
        {
        public:
            static long long get_file_size(const char *name);
        };

        class StackBuffer
        {
        private:
            char arr[1024];
            int idx;

        public:
            StackBuffer();
            ~StackBuffer();

            void append(char c);
            void append(char *s);
            void append(int i);
            void append(float f);
            char *get();
            int get_size();
            void clear();
            bool is_empty();
            bool contains(char c);
            bool is_numeric();
        };

        struct Range
        {
            int beg_idx;
            int end_idx;
        };
    }
}