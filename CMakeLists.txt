cmake_minimum_required( VERSION 3.0 )

set( PROJECT_NAME "NHTTC" )
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -g -Wall -Wextra -Wpedantic -Wno-unused-parameter" )

# Set our project name
project( ${PROJECT_NAME} )

find_package(Threads REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# include_directories( SYSTEM ext )
# include_directories( SYSTEM ext/eigen )
# include_directories( SYSTEM ext/glm )
# include_directories( SYSTEM ext/glad/include )
# include_directories( SYSTEM ext/glfw/include )

# add_subdirectory( ext )

add_library(nativefd STATIC IMPORTED)
set_target_properties(nativefd PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/ext/nativefd/lib/libnfd_linux.a)

include_directories(src)
add_subdirectory(src)
