#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
uint32_t asterion_driver_abi_version(void);
int asterion_driver_open(const unsigned char* cubin, size_t size, const char* entry,
                         uint32_t shared_bytes, uint32_t threads, uint32_t argument_count,
                         void** output, char* error, size_t error_size);
int asterion_driver_launch(void* handle, const uint64_t* values, size_t count,
                           uint32_t grid_x, uint32_t grid_y, void* stream,
                           char* error, size_t error_size);
// Caller serializes handle lifetime. Waits per-stream launch events, or the borrowed
// context after graph capture. Never destroys that context. No graph replay after close.
int asterion_driver_close(void* handle, char* error, size_t error_size);
#ifdef __cplusplus
}
#endif
