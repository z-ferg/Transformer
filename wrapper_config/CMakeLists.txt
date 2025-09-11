cmake_minimum_required(VERSION 3.15)
project(tokenizer)

find_package(Python3 REQUIRED COMPONENTS Development)
find_package(pybind11 REQUIRED)

pybind11_add_module(tokenizer tokenizer.cpp)