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

using namespace std;

type Token = string;
type Pair = uint64_t;
type CountMap = unordered_map<Pair, uint64_t

/*  Function for 
*/
CountMap count_pairs_parallel(){

}

/*  Function for finding the best pair among all 
*/
Pair find_best_pair(unordered_map<Pair, uint64_t>& counts){

}

/*  
*/
tokenize(vector<vector<uint32_t>>& corpus){

}


PYBIND11_MODULE(tokenizer, m) {
    m.doc() = "Parallelized byte pair encoding tokenizer via PyBind11";

    m.def("tokenize", &tokenize, "A function for tokenizing text",
        pybind11::arg("vocab"), pybind11::arg("text"), pybind11::arg("max_vocab_size"), pybind11::arg("specials"));
}