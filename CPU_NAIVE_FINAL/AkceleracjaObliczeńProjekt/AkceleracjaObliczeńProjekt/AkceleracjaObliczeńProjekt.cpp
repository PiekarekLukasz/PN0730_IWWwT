// AkceleracjaObliczeńProjekt.cpp : Ten plik zawiera funkcję „main”. W nim rozpoczyna się i kończy wykonywanie programu.
//

#include <iostream>
#include <fstream>
#include <windows.h>
#include <string.h>

#include <tuple>
#include <set>
#include <vector>

#include <thread>
#include <mutex>

using namespace std;

#define BASE_CHUNK_SIZE 1024
#define CHUNK_SIZE_MULTIPLIER 1 
#define CHUNK_SIZE (BASE_CHUNK_SIZE*CHUNK_SIZE_MULTIPLIER)  // tego też nie dodykać, twarda definicja
#define MAX_THREADS 4

#define MAX_WORD_LENGHT 64
#define MAX_PATH_LENGHT 300

vector<thread> threads;
int currentThreads = 0;
mutex mainListMutex;

int searchDirectoryWithLake(const char* path, const char* word,
    std::vector<std::tuple<char*, std::set<int>>>& files_and_hits);

int readFiletoBuffer(char const* path, vector<char*>& chunks)
{
    FILE* f = fopen(path, "rb");
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);

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

bool isTxtFile(char const* filename)
{
    int fnLen = strlen(filename);
    if (fnLen < 5) return false;
    char fExtension[5];
    strncpy(fExtension, filename + fnLen - 4, 4);
    fExtension[4] = '\0';
    return strcmp(fExtension, ".txt") == 0 ? true : false;
}

int findAllFilesInDir(char const* path, vector<char*>& files)
{
    char expandedpath[MAX_PATH_LENGHT];
    strcpy(expandedpath, path);
    strcat(expandedpath, "\\*");

    WIN32_FIND_DATA data;
    HANDLE hFIND = FindFirstFile(expandedpath, &data);

    if (hFIND == INVALID_HANDLE_VALUE) return 1;
    do {
        if (((data.dwFileAttributes | FILE_ATTRIBUTE_DIRECTORY) == FILE_ATTRIBUTE_DIRECTORY) && (data.cFileName[0] != '.'))
        {
            char* fullpath = (char*)calloc(MAX_PATH_LENGHT, sizeof(char));
            strcpy(fullpath, path);
            strcat(fullpath, "\\");
            strcat(fullpath, data.cFileName);
            findAllFilesInDir(fullpath, files);
            free(fullpath);
        }
        else if (isTxtFile(data.cFileName))
        {
            char* fullpath = (char*)calloc(MAX_PATH_LENGHT, sizeof(char));
            strcpy(fullpath, path);
            strcat(fullpath, "\\");
            strcat(fullpath, data.cFileName);
            files.push_back(fullpath);
        }
    } while (FindNextFile(hFIND, &data));

    return 0;
}

int printResults(vector<tuple<char*, set<int>>>& files_and_hits)
{
    for (tuple<char*, set<int>> file_hits : files_and_hits)
    {
        printf("In file: %s \n", get<0>(file_hits));

        if (get<1>(file_hits).size() == 0) printf("None found! \n");
        else for (int result : get<1>(file_hits)) printf("Hit on pos: %d \n", result);

        printf("\n");

        free(get<0>(file_hits));
    }
    return 0;
}

void naiveTextSearch(char* chunk, const char* word, const unsigned int wor_len, unsigned int pos_shift, set<int> & results)
{
    set<int> localHits;

    for (unsigned int i = 0; i < (CHUNK_SIZE - wor_len); i++)
    {
        bool found = true;
        for (unsigned int j = 0; j < wor_len; j++)
        {
            if (chunk[i + j] != word[j]) found = false;
        }
        if (found)
        {
            localHits.insert(i + pos_shift);
        }
    }

    mainListMutex.lock();
    free(chunk);
    results.merge(localHits);
    currentThreads--;
    mainListMutex.unlock();
    return;
}

int main()
{
    vector<
        tuple<
        char*, set<int>
        >
    >  files_and_hits;

    char word[MAX_WORD_LENGHT];
    char path[MAX_PATH_LENGHT];


    printf("Path to the folder to be searched: \n");
    scanf("%s", path);
    printf("Word to be searched for: \n");
    scanf("%s", word);
    printf("\n|====|Search resutls|====|\n\n");
    
    searchDirectoryWithLake(path, word, files_and_hits);

    printResults(files_and_hits);

    return 0;  
}

int searchDirectoryWithLake(const char* path, const char* word,
    std::vector<std::tuple<char*, std::set<int>>>& files_and_hits)
{
    vector<char*> files;
    const unsigned int wor_len = strlen(word);

    if (findAllFilesInDir(path, files)) {
        printf("Invalid folder path! \n");
        return 1;
    }

    for (char* file : files)
    {
        unsigned int pos_shift = 1;
        vector<char*> chunks;
        readFiletoBuffer(file, chunks);
        set<int> results;

        for (char* contents : chunks)
        {
            while (currentThreads >= MAX_THREADS);
            mainListMutex.lock();
            currentThreads++;
            mainListMutex.unlock();
            threads.push_back(thread(&naiveTextSearch, contents, word, wor_len, pos_shift, ref(results)));
            pos_shift += (CHUNK_SIZE - MAX_WORD_LENGHT - 1);
        }

        for (thread& leftover : threads)
        {
            leftover.join();
        }
        threads.clear();

        files_and_hits.push_back(tuple<char*, set<int>>(file, results));
    }
    
  
    return 0;
}