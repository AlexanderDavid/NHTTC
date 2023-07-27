find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package (spdlog REQUIRED)

set( INTERFACE_HEADERS util.h agent.h )
set( INTERFACE_SOURCE util.cpp agent.cpp )

add_library(nhttc_interface SHARED ${INTERFACE_SOURCE} ${INTERFACE_HEADERS})

target_link_libraries(nhttc_interface Eigen3::Eigen spdlog::spdlog)