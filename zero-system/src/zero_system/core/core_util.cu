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

void StackBuffer::append(char *s)
{
    sprintf(this->arr, "%s%s", this->arr, s);
    this->idx = strlen(this->arr);
}

void StackBuffer::append(int i)
{
    sprintf(this->arr, "%s%d", this->arr, i);
    this->idx = strlen(this->arr);
}

void StackBuffer::append(float f)
{
    sprintf(this->arr, "%s%f", this->arr, f);
    this->idx = strlen(this->arr);
}

char *StackBuffer::get()
{
    return this->arr;
}

int StackBuffer::get_size()
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
    for (int i = 0; i < this->idx; i++)
    {
        if (isdigit(this->arr[i]) == 0)
        {
            return false;
        }
    }

    return true;
}