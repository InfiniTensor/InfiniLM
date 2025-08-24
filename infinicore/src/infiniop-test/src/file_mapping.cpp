#include "file_mapping.hpp"
#include <stdexcept>

FileMapping::FileMapping(const std::string &filepath) {
#ifdef _WIN32
    _file_handle = CreateFile(filepath.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (_file_handle == INVALID_HANDLE_VALUE) {
        throw std::runtime_error("Failed to open GGUF file");
    }
    _file_mapping = CreateFileMapping(_file_handle, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!_file_mapping) {
        CloseHandle(_file_handle);
        throw std::runtime_error("Failed to create file mapping");
    }
    _ptr = MapViewOfFile(_file_mapping, FILE_MAP_READ, 0, 0, 0);
    if (!_ptr) {
        CloseHandle(_file_mapping);
        CloseHandle(_file_handle);
        throw std::runtime_error("Failed to map view of file");
    }
    _size = GetFileSize(_file_handle, NULL);
#else
    int fd = open(filepath.c_str(), O_RDONLY);
    if (fd == -1) {
        throw std::runtime_error("Failed to open GGUF file");
    }
    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        close(fd);
        throw std::runtime_error("Failed to get file size");
    }
    _size = sb.st_size;
    _ptr = mmap(NULL, _size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (_ptr == MAP_FAILED) {
        throw std::runtime_error("Failed to mmap file");
    }
#endif
}

FileMapping::~FileMapping() {
#ifdef _WIN32
    if (_ptr) {
        UnmapViewOfFile(_ptr);
    }
    if (_file_mapping) {
        CloseHandle(_file_mapping);
    }
    if (_file_handle) {
        CloseHandle(_file_handle);
    }
#else
    if (_ptr) {
        munmap(_ptr, _size);
    }
#endif
}

void *FileMapping::ptr() const {
    return _ptr;
}

size_t FileMapping::size() const {
    return _size;
}
