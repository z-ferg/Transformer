#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <map>
#include <sstream>

using namespace std;

std::vector<std::string> tokenize(
    const std::string &vocab_str,               // List of all vocabulary recognized before (letters, punctuation, numbers)
    const std::string &text,                    // Text to tokenize and add to vocabulary
    const int vocab_size,                       // Set a limit for maximum size of vocabulary
    const std::vector<std::string> &specials    // Vector holding special characters to be recognized
) {
    std::vector<std::string> vocab;

    // Initialize the vocabulary with specials and alpha/punc/digits passed from python
    vocab.insert(vocab.end(), specials.begin(), specials.end());
    for (char c: vocab_str){
        vocab.push_back(std::string(1, c));
    }

    map<string, int> freq_list;

    for(size_t i = 0; i < text.length() - 1; i++) {
        std::string bigram = text.substr(i, 2);
        freq_list[bigram]++;
    }

    vector<pair<string, int>> freq_vec (freq_list.begin(), freq_list.end());

    sort(freq_vec.begin(), freq_vec.end(), [](const pair<string, int> &a, const pair<string, int> &b){
        return a.second > b.second;
    });

    for (auto &p : freq_vec) {
        if (vocab.size() >= vocab_size) break;

        if (find(vocab.begin(), vocab.end(), p.first) == vocab.end()) {
            vocab.push_back(p.first);
        }
    }

    return vocab;
}


PYBIND11_MODULE(tokenizer, m) {
    m.doc() = "Example module created with pybind11";

    m.def("tokenize", &tokenize, "A function for tokenizing text",
        pybind11::arg("vocab"), pybind11::arg("text"), pybind11::arg("max_vocab_size"), pybind11::arg("specials"));
}
