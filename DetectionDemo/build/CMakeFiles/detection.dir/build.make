# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/jason/github/Track_ball/DetectionDemo

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jason/github/Track_ball/DetectionDemo/build

# Include any dependencies generated for this target.
include CMakeFiles/detection.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/detection.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/detection.dir/flags.make

CMakeFiles/detection.dir/detection.cpp.o: CMakeFiles/detection.dir/flags.make
CMakeFiles/detection.dir/detection.cpp.o: ../detection.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jason/github/Track_ball/DetectionDemo/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/detection.dir/detection.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/detection.dir/detection.cpp.o -c /home/jason/github/Track_ball/DetectionDemo/detection.cpp

CMakeFiles/detection.dir/detection.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/detection.dir/detection.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jason/github/Track_ball/DetectionDemo/detection.cpp > CMakeFiles/detection.dir/detection.cpp.i

CMakeFiles/detection.dir/detection.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/detection.dir/detection.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jason/github/Track_ball/DetectionDemo/detection.cpp -o CMakeFiles/detection.dir/detection.cpp.s

CMakeFiles/detection.dir/detection.cpp.o.requires:

.PHONY : CMakeFiles/detection.dir/detection.cpp.o.requires

CMakeFiles/detection.dir/detection.cpp.o.provides: CMakeFiles/detection.dir/detection.cpp.o.requires
	$(MAKE) -f CMakeFiles/detection.dir/build.make CMakeFiles/detection.dir/detection.cpp.o.provides.build
.PHONY : CMakeFiles/detection.dir/detection.cpp.o.provides

CMakeFiles/detection.dir/detection.cpp.o.provides.build: CMakeFiles/detection.dir/detection.cpp.o


# Object files for target detection
detection_OBJECTS = \
"CMakeFiles/detection.dir/detection.cpp.o"

# External object files for target detection
detection_EXTERNAL_OBJECTS =

detection: CMakeFiles/detection.dir/detection.cpp.o
detection: CMakeFiles/detection.dir/build.make
detection: /usr/local/cuda-10.2/lib64/libcudart_static.a
detection: /usr/lib/aarch64-linux-gnu/librt.so
detection: /usr/local/lib/libjetson-inference.so
detection: /usr/local/lib/libjetson-utils.so
detection: /usr/local/cuda/lib64/libcudart_static.a
detection: /usr/lib/aarch64-linux-gnu/librt.so
detection: /usr/local/cuda/lib64/libnppicc.so
detection: CMakeFiles/detection.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/jason/github/Track_ball/DetectionDemo/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable detection"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/detection.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/detection.dir/build: detection

.PHONY : CMakeFiles/detection.dir/build

CMakeFiles/detection.dir/requires: CMakeFiles/detection.dir/detection.cpp.o.requires

.PHONY : CMakeFiles/detection.dir/requires

CMakeFiles/detection.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/detection.dir/cmake_clean.cmake
.PHONY : CMakeFiles/detection.dir/clean

CMakeFiles/detection.dir/depend:
	cd /home/jason/github/Track_ball/DetectionDemo/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jason/github/Track_ball/DetectionDemo /home/jason/github/Track_ball/DetectionDemo /home/jason/github/Track_ball/DetectionDemo/build /home/jason/github/Track_ball/DetectionDemo/build /home/jason/github/Track_ball/DetectionDemo/build/CMakeFiles/detection.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/detection.dir/depend

