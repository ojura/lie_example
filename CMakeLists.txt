cmake_minimum_required(VERSION 3.5.1)

add_subdirectory(manif)

project(lie_example)

find_package(Ceres REQUIRED)
find_package(catkin REQUIRED COMPONENTS roscpp geometry_msgs tf2 tf2_ros)


add_executable(lie_example lie_example.cpp)
target_link_libraries(lie_example ceres manif ${catkin_LIBRARIES})
target_include_directories(lie_example PRIVATE ${catkin_INCLUDE_DIRS})
