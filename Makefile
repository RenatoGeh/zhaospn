# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

# Default target executed when no arguments are given to make.
default_target: all
.PHONY : default_target

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake28

# The command to remove a file.
RM = /usr/bin/cmake28 -E remove -f

# Escaping for special characters.
EQUALS = =

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = /usr/bin/ccmake

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/h73zhao/SPN-EM

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/h73zhao/SPN-EM

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake cache editor..."
	/usr/bin/ccmake -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache
.PHONY : edit_cache/fast

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/usr/bin/cmake28 -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache
.PHONY : rebuild_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /home/h73zhao/SPN-EM/CMakeFiles /home/h73zhao/SPN-EM/CMakeFiles/progress.marks
	$(MAKE) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /home/h73zhao/SPN-EM/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean
.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named batch_learning

# Build rule for target.
batch_learning: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 batch_learning
.PHONY : batch_learning

# fast build rule for target.
batch_learning/fast:
	$(MAKE) -f CMakeFiles/batch_learning.dir/build.make CMakeFiles/batch_learning.dir/build
.PHONY : batch_learning/fast

#=============================================================================
# Target rules for targets named online_learning

# Build rule for target.
online_learning: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 online_learning
.PHONY : online_learning

# fast build rule for target.
online_learning/fast:
	$(MAKE) -f CMakeFiles/online_learning.dir/build.make CMakeFiles/online_learning.dir/build
.PHONY : online_learning/fast

#=============================================================================
# Target rules for targets named stream_learning

# Build rule for target.
stream_learning: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 stream_learning
.PHONY : stream_learning

# fast build rule for target.
stream_learning/fast:
	$(MAKE) -f CMakeFiles/stream_learning.dir/build.make CMakeFiles/stream_learning.dir/build
.PHONY : stream_learning/fast

batch_learning.o: batch_learning.cpp.o
.PHONY : batch_learning.o

# target to build an object file
batch_learning.cpp.o:
	$(MAKE) -f CMakeFiles/batch_learning.dir/build.make CMakeFiles/batch_learning.dir/batch_learning.cpp.o
.PHONY : batch_learning.cpp.o

batch_learning.i: batch_learning.cpp.i
.PHONY : batch_learning.i

# target to preprocess a source file
batch_learning.cpp.i:
	$(MAKE) -f CMakeFiles/batch_learning.dir/build.make CMakeFiles/batch_learning.dir/batch_learning.cpp.i
.PHONY : batch_learning.cpp.i

batch_learning.s: batch_learning.cpp.s
.PHONY : batch_learning.s

# target to generate assembly for a file
batch_learning.cpp.s:
	$(MAKE) -f CMakeFiles/batch_learning.dir/build.make CMakeFiles/batch_learning.dir/batch_learning.cpp.s
.PHONY : batch_learning.cpp.s

online_learning.o: online_learning.cpp.o
.PHONY : online_learning.o

# target to build an object file
online_learning.cpp.o:
	$(MAKE) -f CMakeFiles/online_learning.dir/build.make CMakeFiles/online_learning.dir/online_learning.cpp.o
.PHONY : online_learning.cpp.o

online_learning.i: online_learning.cpp.i
.PHONY : online_learning.i

# target to preprocess a source file
online_learning.cpp.i:
	$(MAKE) -f CMakeFiles/online_learning.dir/build.make CMakeFiles/online_learning.dir/online_learning.cpp.i
.PHONY : online_learning.cpp.i

online_learning.s: online_learning.cpp.s
.PHONY : online_learning.s

# target to generate assembly for a file
online_learning.cpp.s:
	$(MAKE) -f CMakeFiles/online_learning.dir/build.make CMakeFiles/online_learning.dir/online_learning.cpp.s
.PHONY : online_learning.cpp.s

src/BatchParamLearning.o: src/BatchParamLearning.cpp.o
.PHONY : src/BatchParamLearning.o

# target to build an object file
src/BatchParamLearning.cpp.o:
	$(MAKE) -f CMakeFiles/batch_learning.dir/build.make CMakeFiles/batch_learning.dir/src/BatchParamLearning.cpp.o
.PHONY : src/BatchParamLearning.cpp.o

src/BatchParamLearning.i: src/BatchParamLearning.cpp.i
.PHONY : src/BatchParamLearning.i

# target to preprocess a source file
src/BatchParamLearning.cpp.i:
	$(MAKE) -f CMakeFiles/batch_learning.dir/build.make CMakeFiles/batch_learning.dir/src/BatchParamLearning.cpp.i
.PHONY : src/BatchParamLearning.cpp.i

src/BatchParamLearning.s: src/BatchParamLearning.cpp.s
.PHONY : src/BatchParamLearning.s

# target to generate assembly for a file
src/BatchParamLearning.cpp.s:
	$(MAKE) -f CMakeFiles/batch_learning.dir/build.make CMakeFiles/batch_learning.dir/src/BatchParamLearning.cpp.s
.PHONY : src/BatchParamLearning.cpp.s

src/OnlineParamLearning.o: src/OnlineParamLearning.cpp.o
.PHONY : src/OnlineParamLearning.o

# target to build an object file
src/OnlineParamLearning.cpp.o:
	$(MAKE) -f CMakeFiles/online_learning.dir/build.make CMakeFiles/online_learning.dir/src/OnlineParamLearning.cpp.o
.PHONY : src/OnlineParamLearning.cpp.o

src/OnlineParamLearning.i: src/OnlineParamLearning.cpp.i
.PHONY : src/OnlineParamLearning.i

# target to preprocess a source file
src/OnlineParamLearning.cpp.i:
	$(MAKE) -f CMakeFiles/online_learning.dir/build.make CMakeFiles/online_learning.dir/src/OnlineParamLearning.cpp.i
.PHONY : src/OnlineParamLearning.cpp.i

src/OnlineParamLearning.s: src/OnlineParamLearning.cpp.s
.PHONY : src/OnlineParamLearning.s

# target to generate assembly for a file
src/OnlineParamLearning.cpp.s:
	$(MAKE) -f CMakeFiles/online_learning.dir/build.make CMakeFiles/online_learning.dir/src/OnlineParamLearning.cpp.s
.PHONY : src/OnlineParamLearning.cpp.s

src/SPNNode.o: src/SPNNode.cpp.o
.PHONY : src/SPNNode.o

# target to build an object file
src/SPNNode.cpp.o:
	$(MAKE) -f CMakeFiles/batch_learning.dir/build.make CMakeFiles/batch_learning.dir/src/SPNNode.cpp.o
	$(MAKE) -f CMakeFiles/online_learning.dir/build.make CMakeFiles/online_learning.dir/src/SPNNode.cpp.o
	$(MAKE) -f CMakeFiles/stream_learning.dir/build.make CMakeFiles/stream_learning.dir/src/SPNNode.cpp.o
.PHONY : src/SPNNode.cpp.o

src/SPNNode.i: src/SPNNode.cpp.i
.PHONY : src/SPNNode.i

# target to preprocess a source file
src/SPNNode.cpp.i:
	$(MAKE) -f CMakeFiles/batch_learning.dir/build.make CMakeFiles/batch_learning.dir/src/SPNNode.cpp.i
	$(MAKE) -f CMakeFiles/online_learning.dir/build.make CMakeFiles/online_learning.dir/src/SPNNode.cpp.i
	$(MAKE) -f CMakeFiles/stream_learning.dir/build.make CMakeFiles/stream_learning.dir/src/SPNNode.cpp.i
.PHONY : src/SPNNode.cpp.i

src/SPNNode.s: src/SPNNode.cpp.s
.PHONY : src/SPNNode.s

# target to generate assembly for a file
src/SPNNode.cpp.s:
	$(MAKE) -f CMakeFiles/batch_learning.dir/build.make CMakeFiles/batch_learning.dir/src/SPNNode.cpp.s
	$(MAKE) -f CMakeFiles/online_learning.dir/build.make CMakeFiles/online_learning.dir/src/SPNNode.cpp.s
	$(MAKE) -f CMakeFiles/stream_learning.dir/build.make CMakeFiles/stream_learning.dir/src/SPNNode.cpp.s
.PHONY : src/SPNNode.cpp.s

src/SPNetwork.o: src/SPNetwork.cpp.o
.PHONY : src/SPNetwork.o

# target to build an object file
src/SPNetwork.cpp.o:
	$(MAKE) -f CMakeFiles/batch_learning.dir/build.make CMakeFiles/batch_learning.dir/src/SPNetwork.cpp.o
	$(MAKE) -f CMakeFiles/online_learning.dir/build.make CMakeFiles/online_learning.dir/src/SPNetwork.cpp.o
	$(MAKE) -f CMakeFiles/stream_learning.dir/build.make CMakeFiles/stream_learning.dir/src/SPNetwork.cpp.o
.PHONY : src/SPNetwork.cpp.o

src/SPNetwork.i: src/SPNetwork.cpp.i
.PHONY : src/SPNetwork.i

# target to preprocess a source file
src/SPNetwork.cpp.i:
	$(MAKE) -f CMakeFiles/batch_learning.dir/build.make CMakeFiles/batch_learning.dir/src/SPNetwork.cpp.i
	$(MAKE) -f CMakeFiles/online_learning.dir/build.make CMakeFiles/online_learning.dir/src/SPNetwork.cpp.i
	$(MAKE) -f CMakeFiles/stream_learning.dir/build.make CMakeFiles/stream_learning.dir/src/SPNetwork.cpp.i
.PHONY : src/SPNetwork.cpp.i

src/SPNetwork.s: src/SPNetwork.cpp.s
.PHONY : src/SPNetwork.s

# target to generate assembly for a file
src/SPNetwork.cpp.s:
	$(MAKE) -f CMakeFiles/batch_learning.dir/build.make CMakeFiles/batch_learning.dir/src/SPNetwork.cpp.s
	$(MAKE) -f CMakeFiles/online_learning.dir/build.make CMakeFiles/online_learning.dir/src/SPNetwork.cpp.s
	$(MAKE) -f CMakeFiles/stream_learning.dir/build.make CMakeFiles/stream_learning.dir/src/SPNetwork.cpp.s
.PHONY : src/SPNetwork.cpp.s

src/StreamParamLearning.o: src/StreamParamLearning.cpp.o
.PHONY : src/StreamParamLearning.o

# target to build an object file
src/StreamParamLearning.cpp.o:
	$(MAKE) -f CMakeFiles/stream_learning.dir/build.make CMakeFiles/stream_learning.dir/src/StreamParamLearning.cpp.o
.PHONY : src/StreamParamLearning.cpp.o

src/StreamParamLearning.i: src/StreamParamLearning.cpp.i
.PHONY : src/StreamParamLearning.i

# target to preprocess a source file
src/StreamParamLearning.cpp.i:
	$(MAKE) -f CMakeFiles/stream_learning.dir/build.make CMakeFiles/stream_learning.dir/src/StreamParamLearning.cpp.i
.PHONY : src/StreamParamLearning.cpp.i

src/StreamParamLearning.s: src/StreamParamLearning.cpp.s
.PHONY : src/StreamParamLearning.s

# target to generate assembly for a file
src/StreamParamLearning.cpp.s:
	$(MAKE) -f CMakeFiles/stream_learning.dir/build.make CMakeFiles/stream_learning.dir/src/StreamParamLearning.cpp.s
.PHONY : src/StreamParamLearning.cpp.s

src/utils.o: src/utils.cpp.o
.PHONY : src/utils.o

# target to build an object file
src/utils.cpp.o:
	$(MAKE) -f CMakeFiles/batch_learning.dir/build.make CMakeFiles/batch_learning.dir/src/utils.cpp.o
	$(MAKE) -f CMakeFiles/online_learning.dir/build.make CMakeFiles/online_learning.dir/src/utils.cpp.o
	$(MAKE) -f CMakeFiles/stream_learning.dir/build.make CMakeFiles/stream_learning.dir/src/utils.cpp.o
.PHONY : src/utils.cpp.o

src/utils.i: src/utils.cpp.i
.PHONY : src/utils.i

# target to preprocess a source file
src/utils.cpp.i:
	$(MAKE) -f CMakeFiles/batch_learning.dir/build.make CMakeFiles/batch_learning.dir/src/utils.cpp.i
	$(MAKE) -f CMakeFiles/online_learning.dir/build.make CMakeFiles/online_learning.dir/src/utils.cpp.i
	$(MAKE) -f CMakeFiles/stream_learning.dir/build.make CMakeFiles/stream_learning.dir/src/utils.cpp.i
.PHONY : src/utils.cpp.i

src/utils.s: src/utils.cpp.s
.PHONY : src/utils.s

# target to generate assembly for a file
src/utils.cpp.s:
	$(MAKE) -f CMakeFiles/batch_learning.dir/build.make CMakeFiles/batch_learning.dir/src/utils.cpp.s
	$(MAKE) -f CMakeFiles/online_learning.dir/build.make CMakeFiles/online_learning.dir/src/utils.cpp.s
	$(MAKE) -f CMakeFiles/stream_learning.dir/build.make CMakeFiles/stream_learning.dir/src/utils.cpp.s
.PHONY : src/utils.cpp.s

stream_learning.o: stream_learning.cpp.o
.PHONY : stream_learning.o

# target to build an object file
stream_learning.cpp.o:
	$(MAKE) -f CMakeFiles/stream_learning.dir/build.make CMakeFiles/stream_learning.dir/stream_learning.cpp.o
.PHONY : stream_learning.cpp.o

stream_learning.i: stream_learning.cpp.i
.PHONY : stream_learning.i

# target to preprocess a source file
stream_learning.cpp.i:
	$(MAKE) -f CMakeFiles/stream_learning.dir/build.make CMakeFiles/stream_learning.dir/stream_learning.cpp.i
.PHONY : stream_learning.cpp.i

stream_learning.s: stream_learning.cpp.s
.PHONY : stream_learning.s

# target to generate assembly for a file
stream_learning.cpp.s:
	$(MAKE) -f CMakeFiles/stream_learning.dir/build.make CMakeFiles/stream_learning.dir/stream_learning.cpp.s
.PHONY : stream_learning.cpp.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... batch_learning"
	@echo "... edit_cache"
	@echo "... online_learning"
	@echo "... rebuild_cache"
	@echo "... stream_learning"
	@echo "... batch_learning.o"
	@echo "... batch_learning.i"
	@echo "... batch_learning.s"
	@echo "... online_learning.o"
	@echo "... online_learning.i"
	@echo "... online_learning.s"
	@echo "... src/BatchParamLearning.o"
	@echo "... src/BatchParamLearning.i"
	@echo "... src/BatchParamLearning.s"
	@echo "... src/OnlineParamLearning.o"
	@echo "... src/OnlineParamLearning.i"
	@echo "... src/OnlineParamLearning.s"
	@echo "... src/SPNNode.o"
	@echo "... src/SPNNode.i"
	@echo "... src/SPNNode.s"
	@echo "... src/SPNetwork.o"
	@echo "... src/SPNetwork.i"
	@echo "... src/SPNetwork.s"
	@echo "... src/StreamParamLearning.o"
	@echo "... src/StreamParamLearning.i"
	@echo "... src/StreamParamLearning.s"
	@echo "... src/utils.o"
	@echo "... src/utils.i"
	@echo "... src/utils.s"
	@echo "... stream_learning.o"
	@echo "... stream_learning.i"
	@echo "... stream_learning.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system

