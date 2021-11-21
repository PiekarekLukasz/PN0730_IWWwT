// AkceleracjaObliczeńProjekt.cpp : Ten plik zawiera funkcję „main”. W nim rozpoczyna się i kończy wykonywanie programu.
//

#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <tuple>
#include <vector>
#include <thread>
#include <future>

using namespace std;

vector hits = vector<tuple<string,vector<tuple<int,int>>>>();

vector sufixTable = vector<int>();

string phrase= "";
int phraseLenght = 0;

vector threads = vector<thread>();
int currentThreads = 0;
int maxThreads = 8;
mutex mainListMutex;

void printHits()
{
    for (tuple<string, vector<tuple<int, int>>> hitList : hits)
    {
        vector<tuple<int, int>> posList = get<1>(hitList);
        if (posList.size() != 0)
        {
            cout << get<0>(hitList) << endl;

            for (tuple<int, int> hit : posList)
            {
                cout << " w linii: " << get<0>(hit) << " na pozycji " << get<1>(hit) << endl;
            }
        }
    }
}

void naiveTextSearch(filesystem::path filepath)
{
    vector localHits = vector<tuple<int, int>>();

    setlocale(LC_CTYPE, "Polish");

    string line;
    ifstream myfile;
    myfile.open(filepath);
    if(! myfile.is_open()) cout << "błąd odczytu" << endl;
    while (getline(myfile, line))
    {
        int textLenght = line.length();

        for (int i = 0; i < textLenght - phraseLenght; i++)
        {
            int j;
            bool found = true;
            for (j = 0; j < phraseLenght; j++)
            {
                if (line[i + j] != phrase[j]) found = false;
            }
            if (found)
            {
                localHits.push_back(tuple(i, j));
            }
        }
    }
    myfile.close();

    mainListMutex.lock();
        hits.push_back(tuple(string{ filepath.string() }, localHits));
        currentThreads--;
    mainListMutex.unlock();
}


int main()
{
    cout << "Podaj haslo do wyszukania: ";
    cin >> phrase;
    phraseLenght = phrase.length();
    std::string path = "D:\\folder_testowy";

    for (const auto& entry : filesystem::directory_iterator(path))
    {
        while (currentThreads == maxThreads)
        {
            
        }

        filesystem::path file = entry.path();
        if (file.extension() == ".txt")
        {
            threads.push_back(thread(&naiveTextSearch, file));
            mainListMutex.lock();
            currentThreads++;
            mainListMutex.unlock();
        }
    }

    for (thread & leftover : threads)
    {
        leftover.join();
    }

    printHits();
}
