
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#include <vector>

#define BASE_CHUNK_SIZE 1024  // tego proszę nie zmieniać 
#define CHUNK_SIZE_MULTIPLIER 32 // to kręcić jak dusza zapragnie (od 1 do 2^32 czy coś)
#define CHUNK_SIZE (BASE_CHUNK_SIZE*CHUNK_SIZE_MULTIPLIER)  // tego też nie dodykać, twarda definicja

#define MAX_WORD_LENGHT 64
#define MAX_PATH_LENGHT 300

cudaError_t searchFileWithCuda(std::vector<char*> analyze, const char* word, std::vector<int*>& result, unsigned int word_lenght);

__global__ void searchKernel(const char* analyze, const char* word, int* result, unsigned int word_lenght)
{
    int x = blockIdx.x;
    int i = threadIdx.x;
    if (i < CHUNK_SIZE - word_lenght)
    {
        int j;
        bool found = true;
        for (j = 0; j < word_lenght; j++)
        {
            if (analyze[(BASE_CHUNK_SIZE * x) + i + j] != word[j]) found = false;
        }
        if (found)
        {
            result[(BASE_CHUNK_SIZE * x) + i] = 1;
        }
    }
}

int readFiletoBuffer(char const* path, std::vector<char*>& chunks)
{

    FILE* f = fopen(path, "rb");
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);  /* same as rewind(f); */

    while (fsize > 0)
    {
        long size_to_read = CHUNK_SIZE < fsize ? CHUNK_SIZE : fsize;
        int rollback = -MAX_WORD_LENGHT;

        char* contents = (char*)calloc(CHUNK_SIZE, sizeof(char));
        fread(contents, 1, size_to_read - 1, f);

        contents[size_to_read - 1] = '\0';
        chunks.push_back(contents);

        fseek(f, rollback, SEEK_CUR);
        fsize -= (CHUNK_SIZE + rollback - 1);
    }


    fclose(f);

    return fsize;
}

int findAllFilesInDir(char const* path, std::vector<char*>& files)
{
    char expandedpath[MAX_PATH_LENGHT];
    strcpy(expandedpath, path);
    strcat(expandedpath, "\\*.txt");

    WIN32_FIND_DATA data;
    HANDLE hFIND = FindFirstFile(expandedpath, &data);
    if (hFIND == INVALID_HANDLE_VALUE)
    {
        printf("Invalid folder path!");
        return 1;
    }
    do {
        char* fullpath = (char*)calloc(MAX_PATH_LENGHT, sizeof(char));
        strcpy(fullpath, path);
        strcat(fullpath, "\\");
        strcat(fullpath, data.cFileName);
        files.push_back(fullpath);
    } while (FindNextFile(hFIND, &data));

    return 0;
}

int main()
{
    std::vector<char*> files;

    char word[MAX_WORD_LENGHT];
    int word_lenght;

    char path[MAX_PATH_LENGHT];

    printf("Path to the folder to be searched: \n");
    scanf("%s", path);
    printf("Word to be searched for: \n");
    scanf("%s", word);

    word_lenght = strlen(word);

    findAllFilesInDir(path, files);

    for (char* file : files)
    {
        long pos_shift = 0;
        printf("FILE: %s \n", file);
        std::vector<char*> chunks;
        readFiletoBuffer(file, chunks);
        std::vector<int*> results;

        cudaError_t cudaStatus = searchFileWithCuda(chunks, word, results, word_lenght);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "addWithCuda failed!");
            return 1;
        }


        for (int* result : results)
        {
            for (int i = 0; i < CHUNK_SIZE; i++) if (result[i] == 1) printf("Hit on pos: %d \n", pos_shift + i);
            pos_shift += (CHUNK_SIZE - MAX_WORD_LENGHT - 1);
            free(result);
        }

        free(file);
    }


    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t searchFileWithCuda(std::vector<char*> analyze, const char* word, std::vector<int*>& result, unsigned int word_lenght)
{
    char* internal_analyze = 0;
    int* internal_result = 0;
    char* Internal_word = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&internal_result, CHUNK_SIZE * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&internal_analyze, CHUNK_SIZE * sizeof(char));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&Internal_word, word_lenght * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(Internal_word, word, word_lenght * sizeof(char), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed! third");
        goto Error;
    }


    for (char* contents : analyze)
    {
        cudaStatus = cudaMemcpy(internal_analyze, contents, CHUNK_SIZE * sizeof(char), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed! first");
            goto Error;
        }

        cudaStatus = cudaMemset(internal_result, 0, CHUNK_SIZE * sizeof(int));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed! second");
            goto Error;
        }

        // Launch a kernel on the GPU with one thread for each element.
        searchKernel << < CHUNK_SIZE_MULTIPLIER, BASE_CHUNK_SIZE >> > (internal_analyze, Internal_word, internal_result, word_lenght);

        free(contents);

        // Check for any errors launching the kernel
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }

        // cudaDeviceSynchronize waits for the kernel to finish, and returns
        // any errors encountered during the launch.
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
            goto Error;
        }

        int* partial_result = (int*)calloc(CHUNK_SIZE, sizeof(int));

        // Copy output vector from GPU buffer to host memory.
        cudaStatus = cudaMemcpy(partial_result, internal_result, CHUNK_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }

        result.push_back(partial_result);
    }


Error:
    cudaFree(Internal_word);
    cudaFree(internal_analyze);
    cudaFree(internal_result);

    return cudaStatus;
}
