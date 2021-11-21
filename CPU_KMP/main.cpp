// AkceleracjaObliczeńProjekt.cpp : This file contains entry point of application.

#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <tuple>
#include <vector>
#include <thread>
#include <future>

//#define NAIVE

using namespace std;

vector hits = vector<tuple<string, vector<tuple<int, int>>>>();

string phrase = "";
int phraseLenght = 0;

vector threads = vector<thread>();
int currentThreads = 0;
int maxThreads = 8;
mutex mainListMutex;

void Pref(vector<int>& P, string& S)
{
    unsigned int t = 0, i, n = S.size();
    P.resize(n + 1, 0);

    for (i = 2; i < n; i++)
    {
        while (t > 0 && S[t + 1] != S[i]) t = P[t];
        if (S[t + 1] == S[i]) t++;
        P[i] = t;
    }
}

void KMPSearch(filesystem::path filepath, string line)
{
    vector localHits = vector<tuple<int, int>>();

    int textLenght = line.length();

    string S = "#" + phrase + "#" + line;

    vector<int> P;
    Pref(P, S);

    unsigned int i, ws = phrase.size();

    for (i = ws + 2; i < S.size(); i++)
    {
        if (P[i] == ws)
        {
            int startPosition = i - ws - ws;
            int endPosition = i - ws;
            localHits.push_back(tuple(startPosition, endPosition));
        }
    }

    mainListMutex.lock();
    if (!localHits.empty()) {
        hits.push_back(tuple(string{ filepath.string() }, localHits));
    }
    currentThreads--;
    mainListMutex.unlock();
}

void printHits()
{
    if (hits.empty()) {
        cout << "Word has not been found.";
        return;
    }

    for (tuple<string, vector<tuple<int, int>>> hitList : hits)
    {
        vector<tuple<int, int>> posList = get<1>(hitList);
        if (posList.size() != 0)
        {
            cout << get<0>(hitList) << endl;

            for (tuple<int, int> hit : posList)
            {
                cout << "Starting position: " << get<0>(hit) << " end position " << get<1>(hit) << endl;
            }
        }
    }
}

void naiveTextSearch(filesystem::path filepath, string line)
{
    vector localHits = vector<tuple<int, int>>();

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

    mainListMutex.lock();
    hits.push_back(tuple(string{ filepath.string() }, localHits));
    currentThreads--;
    mainListMutex.unlock();
}

int main()
{
    cout << "Word to find: ";
    cin >> phrase;
    phraseLenght = phrase.length();

    std::string path = "E:\\Repositories\\PN0730_IWWwT\\CPU_KMP";

    setlocale(LC_CTYPE, "Polish");


    for (const auto& entry : filesystem::directory_iterator(path))
    {
        filesystem::path file = entry.path();
        if (file.extension() != ".txt")
        {
            continue;
        }

        string line;
        ifstream myfile;
        myfile.open(file);

        if (!myfile.is_open()) cout << "Some problems occurs during opening file" << endl;

        while (getline(myfile, line))
        {
            while (currentThreads == maxThreads)
            {

            }
#ifdef NAIVE
            threads.push_back(thread(&naiveTextSearch, file, line));
#else
            threads.push_back(thread(&KMPSearch, file, line));
#endif
            mainListMutex.lock();
            currentThreads++;
            mainListMutex.unlock();
        }

        myfile.close();
    }

    for (thread& leftover : threads)
    {
        leftover.join();
    }

    printHits();
}
