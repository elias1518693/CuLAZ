cmake_minimum_required(VERSION 3.16)
set(CMAKE_SUPPRESS_REGENERATION true)

project(LASzip LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# LASzip 3.5.0 (https://github.com/LASzip/LASzip), Apache-2.0, see COPYING.txt
file(GLOB source_files
	"src/*.hpp"
	"src/*.cpp"
	"dll/*.h"
	"include/laszip/*.h"
)

# not part of upstream's library sources
list(FILTER source_files EXCLUDE REGEX ".*src/las(un)?zipper\\.[ch]pp$")

add_library(laszip SHARED ${source_files})

target_include_directories(laszip
	PUBLIC
	${CMAKE_CURRENT_SOURCE_DIR}/dll
	${CMAKE_CURRENT_SOURCE_DIR}/include
	${CMAKE_CURRENT_SOURCE_DIR}/include/laszip
	PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}/src
)

# laszip_dll.cpp defines LASZIP_DYN_LINK and LASZIP_SOURCE itself to export
# the C API; consumers need LASZIP_DYN_LINK to import it
target_compile_definitions(laszip
	INTERFACE
	LASZIP_DYN_LINK
	PRIVATE
	LASZIPDLL_EXPORTS
	UNORDERED
	HAVE_UNORDERED_MAP=1
	NOMINMAX
	WIN32_LEAN_AND_MEAN
	_CRT_SECURE_NO_DEPRECATE
	_CRT_SECURE_NO_WARNINGS
	_CRT_NONSTDC_NO_WARNING
	_SCL_SECURE_NO_WARNINGS
)

set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
