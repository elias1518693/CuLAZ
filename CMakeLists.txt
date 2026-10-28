cmake_minimum_required(VERSION 3.20)

set(TARGET_NAME CuLAZ)
project(${TARGET_NAME} LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# --------------------------------------------------------
# Source files (C++ and CUDA)
# --------------------------------------------------------
file(GLOB PROJECT_SOURCES CONFIGURE_DEPENDS
    src/*.[ch]pp
    src/*.cu
)

# --------------------------------------------------------
# Dependencies
# --------------------------------------------------------
add_subdirectory(libs/laszip)

# --------------------------------------------------------
# Target setup
# --------------------------------------------------------
add_executable(${TARGET_NAME} ${PROJECT_SOURCES})

target_link_libraries(${TARGET_NAME}
    PRIVATE
    laszip
)

target_include_directories(${TARGET_NAME} PRIVATE
    ./src
    ./libs
)

set_property(TARGET ${TARGET_NAME} PROPERTY
    CUDA_ARCHITECTURES "75;86;89;90;120"
)

set_target_properties(${TARGET_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

# --------------------------------------------------------
# Windows-specific options
# --------------------------------------------------------
if (WIN32)
    set_property(TARGET ${TARGET_NAME} PROPERTY
        VS_DEBUGGER_WORKING_DIRECTORY "$(SolutionDir).."
    )
endif()

if (MSVC)
    target_compile_options(${TARGET_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
    )
endif()

# --------------------------------------------------------
# Set startup project (Visual Studio)
# --------------------------------------------------------
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${TARGET_NAME})
