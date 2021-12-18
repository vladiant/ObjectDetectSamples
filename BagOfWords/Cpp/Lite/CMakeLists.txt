cmake_minimum_required(VERSION 3.10)

project(BagOfWordsLite)

find_package(OpenCV REQUIRED)

add_executable(
    ${PROJECT_NAME}
    bagofwords_classification.cpp
)

set_target_properties(
    ${PROJECT_NAME}
    PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)

target_link_libraries(
    ${PROJECT_NAME}
    PRIVATE
        ${OpenCV_LIBS}
)
