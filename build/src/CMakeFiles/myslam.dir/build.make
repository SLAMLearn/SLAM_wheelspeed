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
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/lucky/Momenta/momenta_slam

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/lucky/Momenta/momenta_slam/build

# Include any dependencies generated for this target.
include src/CMakeFiles/myslam.dir/depend.make

# Include the progress variables for this target.
include src/CMakeFiles/myslam.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/myslam.dir/flags.make

src/CMakeFiles/myslam.dir/frame.cpp.o: src/CMakeFiles/myslam.dir/flags.make
src/CMakeFiles/myslam.dir/frame.cpp.o: ../src/frame.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lucky/Momenta/momenta_slam/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/CMakeFiles/myslam.dir/frame.cpp.o"
	cd /home/lucky/Momenta/momenta_slam/build/src && g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/myslam.dir/frame.cpp.o -c /home/lucky/Momenta/momenta_slam/src/frame.cpp

src/CMakeFiles/myslam.dir/frame.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/myslam.dir/frame.cpp.i"
	cd /home/lucky/Momenta/momenta_slam/build/src && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lucky/Momenta/momenta_slam/src/frame.cpp > CMakeFiles/myslam.dir/frame.cpp.i

src/CMakeFiles/myslam.dir/frame.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/myslam.dir/frame.cpp.s"
	cd /home/lucky/Momenta/momenta_slam/build/src && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lucky/Momenta/momenta_slam/src/frame.cpp -o CMakeFiles/myslam.dir/frame.cpp.s

src/CMakeFiles/myslam.dir/frame.cpp.o.requires:

.PHONY : src/CMakeFiles/myslam.dir/frame.cpp.o.requires

src/CMakeFiles/myslam.dir/frame.cpp.o.provides: src/CMakeFiles/myslam.dir/frame.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/myslam.dir/build.make src/CMakeFiles/myslam.dir/frame.cpp.o.provides.build
.PHONY : src/CMakeFiles/myslam.dir/frame.cpp.o.provides

src/CMakeFiles/myslam.dir/frame.cpp.o.provides.build: src/CMakeFiles/myslam.dir/frame.cpp.o


src/CMakeFiles/myslam.dir/mappoint.cpp.o: src/CMakeFiles/myslam.dir/flags.make
src/CMakeFiles/myslam.dir/mappoint.cpp.o: ../src/mappoint.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lucky/Momenta/momenta_slam/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object src/CMakeFiles/myslam.dir/mappoint.cpp.o"
	cd /home/lucky/Momenta/momenta_slam/build/src && g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/myslam.dir/mappoint.cpp.o -c /home/lucky/Momenta/momenta_slam/src/mappoint.cpp

src/CMakeFiles/myslam.dir/mappoint.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/myslam.dir/mappoint.cpp.i"
	cd /home/lucky/Momenta/momenta_slam/build/src && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lucky/Momenta/momenta_slam/src/mappoint.cpp > CMakeFiles/myslam.dir/mappoint.cpp.i

src/CMakeFiles/myslam.dir/mappoint.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/myslam.dir/mappoint.cpp.s"
	cd /home/lucky/Momenta/momenta_slam/build/src && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lucky/Momenta/momenta_slam/src/mappoint.cpp -o CMakeFiles/myslam.dir/mappoint.cpp.s

src/CMakeFiles/myslam.dir/mappoint.cpp.o.requires:

.PHONY : src/CMakeFiles/myslam.dir/mappoint.cpp.o.requires

src/CMakeFiles/myslam.dir/mappoint.cpp.o.provides: src/CMakeFiles/myslam.dir/mappoint.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/myslam.dir/build.make src/CMakeFiles/myslam.dir/mappoint.cpp.o.provides.build
.PHONY : src/CMakeFiles/myslam.dir/mappoint.cpp.o.provides

src/CMakeFiles/myslam.dir/mappoint.cpp.o.provides.build: src/CMakeFiles/myslam.dir/mappoint.cpp.o


src/CMakeFiles/myslam.dir/map.cpp.o: src/CMakeFiles/myslam.dir/flags.make
src/CMakeFiles/myslam.dir/map.cpp.o: ../src/map.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lucky/Momenta/momenta_slam/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object src/CMakeFiles/myslam.dir/map.cpp.o"
	cd /home/lucky/Momenta/momenta_slam/build/src && g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/myslam.dir/map.cpp.o -c /home/lucky/Momenta/momenta_slam/src/map.cpp

src/CMakeFiles/myslam.dir/map.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/myslam.dir/map.cpp.i"
	cd /home/lucky/Momenta/momenta_slam/build/src && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lucky/Momenta/momenta_slam/src/map.cpp > CMakeFiles/myslam.dir/map.cpp.i

src/CMakeFiles/myslam.dir/map.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/myslam.dir/map.cpp.s"
	cd /home/lucky/Momenta/momenta_slam/build/src && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lucky/Momenta/momenta_slam/src/map.cpp -o CMakeFiles/myslam.dir/map.cpp.s

src/CMakeFiles/myslam.dir/map.cpp.o.requires:

.PHONY : src/CMakeFiles/myslam.dir/map.cpp.o.requires

src/CMakeFiles/myslam.dir/map.cpp.o.provides: src/CMakeFiles/myslam.dir/map.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/myslam.dir/build.make src/CMakeFiles/myslam.dir/map.cpp.o.provides.build
.PHONY : src/CMakeFiles/myslam.dir/map.cpp.o.provides

src/CMakeFiles/myslam.dir/map.cpp.o.provides.build: src/CMakeFiles/myslam.dir/map.cpp.o


src/CMakeFiles/myslam.dir/camera.cpp.o: src/CMakeFiles/myslam.dir/flags.make
src/CMakeFiles/myslam.dir/camera.cpp.o: ../src/camera.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lucky/Momenta/momenta_slam/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object src/CMakeFiles/myslam.dir/camera.cpp.o"
	cd /home/lucky/Momenta/momenta_slam/build/src && g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/myslam.dir/camera.cpp.o -c /home/lucky/Momenta/momenta_slam/src/camera.cpp

src/CMakeFiles/myslam.dir/camera.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/myslam.dir/camera.cpp.i"
	cd /home/lucky/Momenta/momenta_slam/build/src && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lucky/Momenta/momenta_slam/src/camera.cpp > CMakeFiles/myslam.dir/camera.cpp.i

src/CMakeFiles/myslam.dir/camera.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/myslam.dir/camera.cpp.s"
	cd /home/lucky/Momenta/momenta_slam/build/src && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lucky/Momenta/momenta_slam/src/camera.cpp -o CMakeFiles/myslam.dir/camera.cpp.s

src/CMakeFiles/myslam.dir/camera.cpp.o.requires:

.PHONY : src/CMakeFiles/myslam.dir/camera.cpp.o.requires

src/CMakeFiles/myslam.dir/camera.cpp.o.provides: src/CMakeFiles/myslam.dir/camera.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/myslam.dir/build.make src/CMakeFiles/myslam.dir/camera.cpp.o.provides.build
.PHONY : src/CMakeFiles/myslam.dir/camera.cpp.o.provides

src/CMakeFiles/myslam.dir/camera.cpp.o.provides.build: src/CMakeFiles/myslam.dir/camera.cpp.o


src/CMakeFiles/myslam.dir/config.cpp.o: src/CMakeFiles/myslam.dir/flags.make
src/CMakeFiles/myslam.dir/config.cpp.o: ../src/config.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lucky/Momenta/momenta_slam/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object src/CMakeFiles/myslam.dir/config.cpp.o"
	cd /home/lucky/Momenta/momenta_slam/build/src && g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/myslam.dir/config.cpp.o -c /home/lucky/Momenta/momenta_slam/src/config.cpp

src/CMakeFiles/myslam.dir/config.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/myslam.dir/config.cpp.i"
	cd /home/lucky/Momenta/momenta_slam/build/src && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lucky/Momenta/momenta_slam/src/config.cpp > CMakeFiles/myslam.dir/config.cpp.i

src/CMakeFiles/myslam.dir/config.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/myslam.dir/config.cpp.s"
	cd /home/lucky/Momenta/momenta_slam/build/src && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lucky/Momenta/momenta_slam/src/config.cpp -o CMakeFiles/myslam.dir/config.cpp.s

src/CMakeFiles/myslam.dir/config.cpp.o.requires:

.PHONY : src/CMakeFiles/myslam.dir/config.cpp.o.requires

src/CMakeFiles/myslam.dir/config.cpp.o.provides: src/CMakeFiles/myslam.dir/config.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/myslam.dir/build.make src/CMakeFiles/myslam.dir/config.cpp.o.provides.build
.PHONY : src/CMakeFiles/myslam.dir/config.cpp.o.provides

src/CMakeFiles/myslam.dir/config.cpp.o.provides.build: src/CMakeFiles/myslam.dir/config.cpp.o


src/CMakeFiles/myslam.dir/g2o_types.cpp.o: src/CMakeFiles/myslam.dir/flags.make
src/CMakeFiles/myslam.dir/g2o_types.cpp.o: ../src/g2o_types.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lucky/Momenta/momenta_slam/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object src/CMakeFiles/myslam.dir/g2o_types.cpp.o"
	cd /home/lucky/Momenta/momenta_slam/build/src && g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/myslam.dir/g2o_types.cpp.o -c /home/lucky/Momenta/momenta_slam/src/g2o_types.cpp

src/CMakeFiles/myslam.dir/g2o_types.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/myslam.dir/g2o_types.cpp.i"
	cd /home/lucky/Momenta/momenta_slam/build/src && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lucky/Momenta/momenta_slam/src/g2o_types.cpp > CMakeFiles/myslam.dir/g2o_types.cpp.i

src/CMakeFiles/myslam.dir/g2o_types.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/myslam.dir/g2o_types.cpp.s"
	cd /home/lucky/Momenta/momenta_slam/build/src && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lucky/Momenta/momenta_slam/src/g2o_types.cpp -o CMakeFiles/myslam.dir/g2o_types.cpp.s

src/CMakeFiles/myslam.dir/g2o_types.cpp.o.requires:

.PHONY : src/CMakeFiles/myslam.dir/g2o_types.cpp.o.requires

src/CMakeFiles/myslam.dir/g2o_types.cpp.o.provides: src/CMakeFiles/myslam.dir/g2o_types.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/myslam.dir/build.make src/CMakeFiles/myslam.dir/g2o_types.cpp.o.provides.build
.PHONY : src/CMakeFiles/myslam.dir/g2o_types.cpp.o.provides

src/CMakeFiles/myslam.dir/g2o_types.cpp.o.provides.build: src/CMakeFiles/myslam.dir/g2o_types.cpp.o


src/CMakeFiles/myslam.dir/visual_odometry.cpp.o: src/CMakeFiles/myslam.dir/flags.make
src/CMakeFiles/myslam.dir/visual_odometry.cpp.o: ../src/visual_odometry.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lucky/Momenta/momenta_slam/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object src/CMakeFiles/myslam.dir/visual_odometry.cpp.o"
	cd /home/lucky/Momenta/momenta_slam/build/src && g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/myslam.dir/visual_odometry.cpp.o -c /home/lucky/Momenta/momenta_slam/src/visual_odometry.cpp

src/CMakeFiles/myslam.dir/visual_odometry.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/myslam.dir/visual_odometry.cpp.i"
	cd /home/lucky/Momenta/momenta_slam/build/src && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lucky/Momenta/momenta_slam/src/visual_odometry.cpp > CMakeFiles/myslam.dir/visual_odometry.cpp.i

src/CMakeFiles/myslam.dir/visual_odometry.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/myslam.dir/visual_odometry.cpp.s"
	cd /home/lucky/Momenta/momenta_slam/build/src && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lucky/Momenta/momenta_slam/src/visual_odometry.cpp -o CMakeFiles/myslam.dir/visual_odometry.cpp.s

src/CMakeFiles/myslam.dir/visual_odometry.cpp.o.requires:

.PHONY : src/CMakeFiles/myslam.dir/visual_odometry.cpp.o.requires

src/CMakeFiles/myslam.dir/visual_odometry.cpp.o.provides: src/CMakeFiles/myslam.dir/visual_odometry.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/myslam.dir/build.make src/CMakeFiles/myslam.dir/visual_odometry.cpp.o.provides.build
.PHONY : src/CMakeFiles/myslam.dir/visual_odometry.cpp.o.provides

src/CMakeFiles/myslam.dir/visual_odometry.cpp.o.provides.build: src/CMakeFiles/myslam.dir/visual_odometry.cpp.o


# Object files for target myslam
myslam_OBJECTS = \
"CMakeFiles/myslam.dir/frame.cpp.o" \
"CMakeFiles/myslam.dir/mappoint.cpp.o" \
"CMakeFiles/myslam.dir/map.cpp.o" \
"CMakeFiles/myslam.dir/camera.cpp.o" \
"CMakeFiles/myslam.dir/config.cpp.o" \
"CMakeFiles/myslam.dir/g2o_types.cpp.o" \
"CMakeFiles/myslam.dir/visual_odometry.cpp.o"

# External object files for target myslam
myslam_EXTERNAL_OBJECTS =

../lib/libmyslam.so: src/CMakeFiles/myslam.dir/frame.cpp.o
../lib/libmyslam.so: src/CMakeFiles/myslam.dir/mappoint.cpp.o
../lib/libmyslam.so: src/CMakeFiles/myslam.dir/map.cpp.o
../lib/libmyslam.so: src/CMakeFiles/myslam.dir/camera.cpp.o
../lib/libmyslam.so: src/CMakeFiles/myslam.dir/config.cpp.o
../lib/libmyslam.so: src/CMakeFiles/myslam.dir/g2o_types.cpp.o
../lib/libmyslam.so: src/CMakeFiles/myslam.dir/visual_odometry.cpp.o
../lib/libmyslam.so: src/CMakeFiles/myslam.dir/build.make
../lib/libmyslam.so: /usr/local/lib/libceres.a
../lib/libmyslam.so: /usr/local/lib/libopencv_shape.so.3.2.0
../lib/libmyslam.so: /usr/local/lib/libopencv_stitching.so.3.2.0
../lib/libmyslam.so: /usr/local/lib/libopencv_superres.so.3.2.0
../lib/libmyslam.so: /usr/local/lib/libopencv_videostab.so.3.2.0
../lib/libmyslam.so: /usr/local/lib/libopencv_viz.so.3.2.0
../lib/libmyslam.so: /home/lucky/slambook/3rdparty/Sophus/build/libSophus.so
../lib/libmyslam.so: /usr/lib/x86_64-linux-gnu/libglog.so
../lib/libmyslam.so: /usr/lib/x86_64-linux-gnu/libspqr.so
../lib/libmyslam.so: /usr/lib/libtbb.so
../lib/libmyslam.so: /usr/lib/libtbbmalloc.so
../lib/libmyslam.so: /usr/lib/x86_64-linux-gnu/libcholmod.so
../lib/libmyslam.so: /usr/lib/x86_64-linux-gnu/libccolamd.so
../lib/libmyslam.so: /usr/lib/x86_64-linux-gnu/libcamd.so
../lib/libmyslam.so: /usr/lib/x86_64-linux-gnu/libcolamd.so
../lib/libmyslam.so: /usr/lib/x86_64-linux-gnu/libamd.so
../lib/libmyslam.so: /usr/lib/liblapack.so
../lib/libmyslam.so: /usr/lib/libblas.so
../lib/libmyslam.so: /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.a
../lib/libmyslam.so: /usr/lib/x86_64-linux-gnu/librt.so
../lib/libmyslam.so: /usr/lib/liblapack.so
../lib/libmyslam.so: /usr/lib/libblas.so
../lib/libmyslam.so: /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.a
../lib/libmyslam.so: /usr/lib/x86_64-linux-gnu/librt.so
../lib/libmyslam.so: /usr/local/lib/libopencv_objdetect.so.3.2.0
../lib/libmyslam.so: /usr/local/lib/libopencv_calib3d.so.3.2.0
../lib/libmyslam.so: /usr/local/lib/libopencv_features2d.so.3.2.0
../lib/libmyslam.so: /usr/local/lib/libopencv_flann.so.3.2.0
../lib/libmyslam.so: /usr/local/lib/libopencv_highgui.so.3.2.0
../lib/libmyslam.so: /usr/local/lib/libopencv_ml.so.3.2.0
../lib/libmyslam.so: /usr/local/lib/libopencv_photo.so.3.2.0
../lib/libmyslam.so: /usr/local/lib/libopencv_video.so.3.2.0
../lib/libmyslam.so: /usr/local/lib/libopencv_videoio.so.3.2.0
../lib/libmyslam.so: /usr/local/lib/libopencv_imgcodecs.so.3.2.0
../lib/libmyslam.so: /usr/local/lib/libopencv_imgproc.so.3.2.0
../lib/libmyslam.so: /usr/local/lib/libopencv_core.so.3.2.0
../lib/libmyslam.so: src/CMakeFiles/myslam.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/lucky/Momenta/momenta_slam/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Linking CXX shared library ../../lib/libmyslam.so"
	cd /home/lucky/Momenta/momenta_slam/build/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/myslam.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/myslam.dir/build: ../lib/libmyslam.so

.PHONY : src/CMakeFiles/myslam.dir/build

src/CMakeFiles/myslam.dir/requires: src/CMakeFiles/myslam.dir/frame.cpp.o.requires
src/CMakeFiles/myslam.dir/requires: src/CMakeFiles/myslam.dir/mappoint.cpp.o.requires
src/CMakeFiles/myslam.dir/requires: src/CMakeFiles/myslam.dir/map.cpp.o.requires
src/CMakeFiles/myslam.dir/requires: src/CMakeFiles/myslam.dir/camera.cpp.o.requires
src/CMakeFiles/myslam.dir/requires: src/CMakeFiles/myslam.dir/config.cpp.o.requires
src/CMakeFiles/myslam.dir/requires: src/CMakeFiles/myslam.dir/g2o_types.cpp.o.requires
src/CMakeFiles/myslam.dir/requires: src/CMakeFiles/myslam.dir/visual_odometry.cpp.o.requires

.PHONY : src/CMakeFiles/myslam.dir/requires

src/CMakeFiles/myslam.dir/clean:
	cd /home/lucky/Momenta/momenta_slam/build/src && $(CMAKE_COMMAND) -P CMakeFiles/myslam.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/myslam.dir/clean

src/CMakeFiles/myslam.dir/depend:
	cd /home/lucky/Momenta/momenta_slam/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/lucky/Momenta/momenta_slam /home/lucky/Momenta/momenta_slam/src /home/lucky/Momenta/momenta_slam/build /home/lucky/Momenta/momenta_slam/build/src /home/lucky/Momenta/momenta_slam/build/src/CMakeFiles/myslam.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/myslam.dir/depend

