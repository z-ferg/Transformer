#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <map>
#include <sstream>
#include <vector>

#include <pthread.h>
#include <unordered_map>

namespace py = pybind11;
using namespace std;

using Token = uint32_t;
using CountMap = unordered_map<Pair, uint64_t>;

using Pair = uint64_t;
inline Pair pack_pair(uint32_t a, uint32_t b) {
    return (static_cast<uint64_t>(a) << 32 | b);
}
inline uint32_t unpack_first(Pair p)  { return p >> 32; }
inline uint32_t unpack_second(Pair p) { return p & 0xFFFFFFFF;}

struct VocabEntry {
    uint32_t token_id;
    string token;
};

/*  Split the given string corpus into individual chars for a first pass
        Takes in the corpus as a string referenced object
        Returns a vector of uint32_t values representing 
*/
vector<Token> split_corpus(const string& corpus, const vector<VocabEntry>& vocab){
    vector<Token> tokens;
    tokens.reserve(corpus.size());

    const size_t offset = vocab.size() - 256;

    for(unsigned char c : corpus) {
        uint32_t token_id = vocab[offset + c].token_id;
        tokens.push_back(token_id);
    }

    return tokens;
}


/*  Find all pairings in the corpus
*/
CountMap count_pairs_parallel(const vector<Token>& tokens){}


/*  Function for finding the best pair among all counted pairs
*/
Pair find_best_pair(const CountMap& counts){}

/*  Function for taking 2 tokens and merging throughout the corpus
        Takes in the current corpus tokens and all IDs required
        Returns a new vector of tokens with properly merged entries
*/
vector<uint32_t> merge_in_corpus(const vector<uint32_t>& tokens, const uint32_t& id_a, const uint32_t& id_b, const uint32_t& new_id){
    vector<uint32_t> new_tokens;
    new_tokens.reserve(tokens.size())

    for (size_t i = 0; i < tokens.size();) {
        if (i + 1 < tokens.size() && tokens[i] == id_a && tokens[i + 1] == id_b){
            new_tokens.push_back(new_id);
            i += 2;
        }
        else {
            new_tokens.push_back(tokens[i++]);
        }
    }

    return new_tokens;
}


/*  Driver function for tokenizing incoming corpus
        Takes in very large chunk of text from python endpoint
        Returns vector of sequential token IDs (decoding not available?)
*/
vector<uint32_t> tokenize(const string& corpus, const vector<string>& specials, const uint32_t& vocab_size){
    /// -------------------------------------
    //        Initialize Vocabulary
    /// -------------------------------------
    vector<VocabEntry> vocab;
    uint32_t next_id = 0;
    vocab.reserve(256 + specials.size())

    // Add special tokens to the vocabulary
    for(string token : specials){
        vocab.push_back({next_id++, token});
    }

    // Add all raw byte values to vocabulary
    for(int byte = 0; byte < 256; ++byte) {
        string tok(1, static_cast<char>(byte));
        vocab.push_back({next_id++, tok})
    }

    vector<uint32_t> corpus_tokens = split_corpus(corpus, vocab);

    /// -------------------------------------
    //           Primary BPE Loop
    /// -------------------------------------
    while(vocab.size() < vocab_size){
        CountMap all_counts = count_pairs_parallel(corpus_tokens);
        Pair best_pair = find_best_pair(all_counts);

        uint32_t id_a = unpack_first(best_pair);
        uint32_t id_b = unpack_second(best_pair);

        uint32_t merged_id = next_id++;
        string merged = vocab[id_a].token + vocab[id_b].token;
        vocab.push_back({merged_id, merged});

        corpus_tokens.swap(merge_in_corpus(corpus_tokens, id_a, id_b, merged_id));
    }

    return corpus_tokens;
}


PYBIND11_MODULE(tokenizer, m) {
    m.doc() = "Parallelized byte pair encoding tokenizer via PyBind11";

    m.def("tokenize", &tokenize, "A function for tokenizing text", 
        py::arg("corpus"), py::arg("specials"), py::arg("vocab_size"));
}