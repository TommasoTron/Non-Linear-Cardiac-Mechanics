cmake_minimum_required(VERSION 3.12.0)
project(lab02 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED "ON")

# Set default build type to Release.
if(NOT CMAKE_BUILD_TYPE OR "${CMAKE_BUILD_TYPE}" STREQUAL "")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()
message(STATUS)
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS)
if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
  add_definitions(-DBUILD_TYPE_DEBUG)
endif()

# Locate MPI compiler.
find_package(MPI REQUIRED)
set(CMAKE_CXX_COMPILER "${MPI_CXX_COMPILER}")

# Locate deal.II and initialize its variables.
find_package(deal.II 9.3.1 REQUIRED
  HINTS ${DEAL_II_DIR} $ENV{DEAL_II_DIR} $ENV{mkDealiiPrefix})
deal_ii_initialize_cached_variables()

# Add useful compiler flags.
#file(GLOB_RECURSE SOURCE_FILES "src/*.cpp" "src/*.hpp")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wfloat-conversion -Wmissing-braces -Wnon-virtual-dtor -Wfatal-errors -g")

add_library(SystemAssembler STATIC src/cardiac/system_assembler.cpp)
deal_ii_setup_target(SystemAssembler)


add_library(Poisson STATIC src/cardiac/Poisson.cpp)
deal_ii_setup_target(Poisson)

add_library(LVBase STATIC src/cardiac/lv_base.cpp)
deal_ii_setup_target(LVBase)

add_library(CardiacTensorUtils STATIC src/cardiac/tensor_utils.cpp)
deal_ii_setup_target(CardiacTensorUtils)

add_library(CardiacLV STATIC src/cardiac/lv.cpp)
deal_ii_setup_target(CardiacLV)

set(MODELS guccione holzapfel)

foreach(MODEL ${MODELS})

    add_executable(cardiac_${MODEL} src/main_${MODEL}.cpp)
    target_link_libraries(cardiac_${MODEL}
        LVBase Poisson
        CardiacLV CardiacTensorUtils)
    deal_ii_setup_target(cardiac_${MODEL})
endforeach()


add_executable(cardiac src/main.cpp)
target_link_libraries(cardiac
     LVBase
    SystemAssembler  Poisson
    CardiacLV CardiacTensorUtils)
deal_ii_setup_target(cardiac)

add_executable(cardiac_neohooke src/main_neohooke.cpp)
target_link_libraries(cardiac_neohooke LVBase Poisson SystemAssembler CardiacLV CardiacTensorUtils)
deal_ii_setup_target(cardiac_neohooke)
