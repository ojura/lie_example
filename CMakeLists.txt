cmake_minimum_required(VERSION 3.5.1)

add_subdirectory(manif)

project(lie_example)

find_package(Ceres REQUIRED)
find_package(ament_cmake)
find_package(rclcpp)
find_package(nav_msgs)
find_package(geometry_msgs)
find_package(tf2)
find_package(tf2_ros)
find_package(visualization_msgs)


add_executable(lie_example lie_example.cpp)
target_link_libraries(lie_example ceres manif
   ${rclcpp_LIBRARIES}
   ${geometry_msgs_LIBRARIES}
   ${tf2_ros_LIBRARIES}
   ${nav_msgs_LIBRARIES}
   ${visualization_msgs_LIBRARIES}
)
