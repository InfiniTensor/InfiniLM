#ifndef __INFINIOPTEST_FILE_MAPPING_HPP__
#define __INFINIOPTEST_FILE_MAPPING_HPP__

#ifdef _WIN32 // windows
#include <windows.h>
#else // linux
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <cstddef>
#include <memory>
#include <string>

class FileMapping {
private:
    void *_ptr;
    size_t _size;
#ifdef _WIN32
    HANDLE _file_handle = NULL;
    HANDLE _file_mapping = NULL;
#endif
public:
    FileMapping(const std::string &filepath);
    ~FileMapping();
    void *ptr() const;
    size_t size() const;
};
#endif // __INFINIOPTEST_FILE_MAPPING_HPP__
