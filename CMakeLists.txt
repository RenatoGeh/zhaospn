cmake_minimum_required(VERSION 2.8)
project(SPN_EM)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")

## Executable for structure learning algorithms
set(SOURCE_FILES_STRUCTURE st_learn.cpp src/SPNNode.cpp src/SPNNode.h src/utils.h src/utils.cpp
        src/SPNetwork.cpp src/SPNetwork.h ## src/StructureLearning.cpp src/StructureLearning.h
        src/BatchParamLearning.cpp src/BatchParamLearning.h)

# BOOST support
find_package(Boost 1.55.0 COMPONENTS system filesystem thread program_options REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/build/")

add_executable(st_learn ${SOURCE_FILES_STRUCTURE})

# Link libs
target_link_libraries(st_learn ${Boost_LIBRARIES})
