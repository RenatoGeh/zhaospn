cmake_minimum_required(VERSION 2.8)
project(zhaospn)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g")

## Executable for structure learning algorithms
set(SOURCE_FILES_STRUCTURE main.cpp src/SPNNode.cpp src/SPNNode.h src/utils.h src/utils.cpp
        src/SPNetwork.cpp src/SPNetwork.h src/BatchParamLearning.cpp src/BatchParamLearning.h
        src/OnlineParamLearning.cpp src/OnlineParamLearning.h src/StreamParamLearning.cpp
        src/StreamParamLearning.h)

# BOOST support
find_package(Boost 1.55.0 COMPONENTS system filesystem thread program_options REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/build/")

#add_executable(main ${SOURCE_FILES_STRUCTURE})

# Link libs
#target_link_libraries(main ${Boost_LIBRARIES})

# Add Python module
find_package(pybind11 REQUIRED)
# Glob all bindings
file(GLOB bindings "bindings/*.h" "bindings/*.cpp")
pybind11_add_module(zhaospn zhaospn.cpp ${bindings} ${SOURCE_FILES_STRUCTURE})
target_link_libraries(zhaospn PRIVATE ${Boost_LIBRARIES})
