cmake_minimum_required(VERSION 3.18)
project(infinirt_consumer_example LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(InfiniRT CONFIG REQUIRED)

add_executable(infinirt_consumer main.cc)
target_link_libraries(infinirt_consumer PRIVATE InfiniRT::infinirt)

