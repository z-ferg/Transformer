#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <map>
#include <sstream>

using namespace std;


vector<pair<pair<string, string>, int>> get_frequency ( const vector<string>& tokens ) {
    map<pair<string, string>, int> ngrams;

    for(size_t i = 0; i < tokens.size() - 1; i++) {
        pair<string, string> ngram = make_pair(tokens[i], tokens[i + 1]);
        ngrams[ngram]++;
    }

    vector<pair<pair<string, string>, int>> ngram_vec(ngrams.begin(), ngrams.end());

    sort(ngram_vec.begin(), ngram_vec.end(), [](const auto &a, const auto &b) {
        return a.second > b.second;
    });

    return ngram_vec;
}


vector<string> split_text (const string& text) {
    vector<string> tokens;
    for (char c : text) {
        string char_to_str = string(1, c);
        tokens.push_back(char_to_str);
    }
    return tokens;
}


std::vector<std::string> tokenize(
    const std::string &vocab_str,               // List of all vocabulary recognized before (letters, punctuation, numbers)
    const std::string &text,                    // Text to tokenize and add to vocabulary
    const size_t vocab_size,                       // Set a limit for maximum size of vocabulary
    const std::vector<std::string> &specials    // Vector holding special characters to be recognized
) {
    vector<string> vocab;

    // Initialize the vocabulary with specials and alpha/punc/digits passed from python
    vocab.insert(vocab.end(), specials.begin(), specials.end());
    for (char c: vocab_str){
        vocab.push_back(std::string(1, c));
    }

    // Tokenize the given text
    vector<string> tokens = split_text(text);
    
    while (vocab.size() <= vocab_size) {
        auto frequency = get_frequency(tokens);
        if (frequency.empty()) break;

        pair<string, string> best_ngram = make_pair(frequency[0].first.first, frequency[0].first.second);
        vocab.push_back(best_ngram.first + best_ngram.second);

        vector<string> new_tokens;
        size_t i = 0;

        while (i < tokens.size()) {
            if (tokens[i] == best_ngram.first and tokens[i + 1] == best_ngram.second) {
                new_tokens.push_back(best_ngram.first + best_ngram.second);
                i += 2;
            }
            else {
                new_tokens.push_back(tokens[i]);
                i += 1;
            }
        }

        tokens = new_tokens;
    }

    return vocab;
}


PYBIND11_MODULE(tokenizer, m) {
    m.doc() = "Example module created with pybind11";

    m.def("tokenize", &tokenize, "A function for tokenizing text",
        pybind11::arg("vocab"), pybind11::arg("text"), pybind11::arg("max_vocab_size"), pybind11::arg("specials"));
}