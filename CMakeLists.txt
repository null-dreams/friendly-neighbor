cmake_minimum_required(VERSION 3.12)
project(friendly_neighbor CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find or download SQLite headers. 
find_path(SQLITE3_INCLUDE_DIR sqlite3.h)
if(NOT SQLITE3_INCLUDE_DIR)
    message(FATAL_ERROR "sqlite3.h not found. Please install sqlite3 development headers.")
endif()

add_library(friendly_neighbor SHARED
    src/distance.cpp
    src/extension.cpp
)

target_include_directories(friendly_neighbor PRIVATE 
    ${SQLITE3_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
set_target_properties(friendly_neighbor PROPERTIES PREFIX "")