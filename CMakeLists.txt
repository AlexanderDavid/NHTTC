cmake_minimum_required(VERSION 3.0.2)
project(nhttc_ros)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  nav_msgs
  geometry_msgs
  visualization_msgs
  tf2
)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)


###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
 INCLUDE_DIRS 
  src/sgd
  src/nhttc_interface
 CATKIN_DEPENDS 
	roscpp 
	nav_msgs
	geometry_msgs
  visualization_msgs
 	tf2
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
  src/nhttc_interface
  src
)

## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
add_executable(${PROJECT_NAME}_node src/nhttc_node.cpp)

## Add cmake target dependencies of the executable
## same as for the library above
add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

add_subdirectory(src/nhttc_interface)
add_subdirectory(src/sgd)
# add_library(nhttc_interface 
#   src/nhttc_interface/nhttc_interface.cpp
#   src/nhttc_interface/util.cpp
# )

# add_library(sgd 
#   src/sgd/ttc_sgd_problem_models.cpp
#   src/sgd/ttc_sgd_problem.cpp
#   src/sgd/ttc_obstacles.cpp
#   src/sgd/sgd.cpp
# )


## Specify libraries to link a library or executable target against
target_link_libraries(${PROJECT_NAME}_node
  ${catkin_LIBRARIES}
  nhttc_interface
  sgd
)