#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <string>
#include <iostream>

int add(int a, int b) {
    return a + b;
}

int sub(int a, int b) {
    return a - b;
}

std::vector<std::string> tokenize(
    std::vector<std::string> vocab,
    std::vector<std::string> text
) {
    std::vector<std::string> fruits = {
        "Grape",
        "Mango",
        "Strawberry"
    };
    return fruits;
}

PYBIND11_MODULE(tokenizer, m) {
    m.doc() = "Example module created with pybind11";
    
    m.def("tokenize", &tokenize, "A function for tokenizing text",
        pybind11::arg("vocab"), pybind11::arg("text"));
}
