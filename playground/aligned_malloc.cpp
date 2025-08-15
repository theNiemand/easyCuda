#include <cstdlib>
#include <iostream>


// use "malloc" to allocate a memory block with a specific alignment
// no other library is allowed to use
void* aligned_malloc(size_t size, size_t alignment) {
    // we need to allocate some space for the original pointer
    // to make sure we can free the memory block correctly
    size_t original_ptr_space = sizeof(void*);

    // malloc extra space for the original pointer and the might needed offset space for alignment
    void* ptr = malloc(size + original_ptr_space + alignment - 1);

    // if malloc failed, return nullptr
    if (ptr == nullptr) {
        return nullptr;
    }

    // First, we need to offset the pointer by the original pointer space
    // to make sure we can store the original pointer
    size_t offset_ptr_space = original_ptr_space;
    void* _ptr = (void*)((size_t)ptr + offset_ptr_space);
    
    // then, we need to offset the pointer by the alignment
    // to make sure the pointer is aligned
    size_t align_remainder = ((size_t)_ptr % alignment);
    size_t offset_alignment = align_remainder ? alignment - align_remainder : 0;
    void* aligned_ptr = (void*)((size_t)_ptr + offset_alignment);

    // store the original pointer just before the aligned pointer
    ((void**)aligned_ptr)[-1] = ptr;

    return aligned_ptr;
    
}

void aligned_free(void* ptr) {
    // get the original pointer from the aligned pointer
    void* original_ptr = ((void**)ptr)[-1];

    // free the whole memory block by the original pointer
    // we got the address return by "malloc" and we can use it.
    // cause there is metadata stored just before the aligned pointer
    // so the "free" can correctly get the metadata.
    free(original_ptr);
}


int main () {
    void* ptr = aligned_malloc(1024, 16);
    if (ptr == nullptr) {
        std::cerr << "malloc failed" << std::endl;
        return 1;
    } else if (((size_t)ptr % 16) != 0) {
        std::cerr << "aligned malloc failed" << std::endl;
        return 1;
    } else {
        std::cout << "aligned malloc success" << std::endl;
    }

    aligned_free(ptr);

    return 0;
}