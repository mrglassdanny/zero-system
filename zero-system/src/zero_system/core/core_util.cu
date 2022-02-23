#include "core_util.cuh"

using namespace zero::core;

long long FileUtils::get_file_size(const char *name)
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

StackBuffer::StackBuffer()
{
    memset(this->arr, 0, sizeof(this->arr));
    this->idx = 0;
}

StackBuffer::~StackBuffer() {}

void StackBuffer::append(char c)
{
    this->arr[this->idx++] = c;
}

char *StackBuffer::get()
{
    return this->arr;
}

int StackBuffer::get_idx()
{
    return this->idx;
}

void StackBuffer::clear()
{
    memset(this->arr, 0, sizeof(char) * this->idx);
    this->idx = 0;
}

bool StackBuffer::is_empty()
{
    return this->idx == 0;
}

bool StackBuffer::contains(char c)
{
    for (int i = 0; i < this->idx; i++)
    {
        if (this->arr[i] == c)
        {
            return true;
        }
    }

    return false;
}

bool StackBuffer::is_numeric()
{
    double val = atof(this->arr);

    if (val == 0.0)
    {
        for (int i = 0; i < this->idx; i++)
        {
            if (this->arr[i] != '0' && this->arr[i] != '.' && this->arr[i] != '-')
            {
                return false;
            }
        }

        return true;
    }
    else
    {
        return true;
    }
}