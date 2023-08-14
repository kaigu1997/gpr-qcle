cmake_minimum_required(VERSION 3.14)

project(gpr LANGUAGES CXX)

# set standard
set(CMAKE_CXX_STANDARD 20) # requires 3.12
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

# find packages
find_package(TBB REQUIRED CONFIG) # required in parallel stl
message(STATUS "Found TBB version ${TBB_VERSION}")

set(MKL_ARCH intel64)
set(MKL_LINK dynamic)
set(MKL_THREADING tbb_thread)
set(MKL_INTERFACE lp64)
set(MKL_MPI intelmpi)
find_package(MKL REQUIRED CONFIG)
message(STATUS "Found MKL version ${MKL_VERSION}")

find_package(Eigen3 3.4.0 REQUIRED CONFIG)
message(STATUS "Found Eigen3 version ${Eigen3_VERSION}")

find_package(xtl REQUIRED CONFIG)
message(STATUS "Found xtl version ${xtl_VERSION}")

set(XTENSOR_USE_TBB 1)
set(XTENSOR_DISABLE_EXCEPTIONS 1)
find_package(xtensor REQUIRED CONFIG)
message(STATUS "Found xtensor version ${xtensor_VERSION}")

find_package(Python3 REQUIRED COMPONENTS Interpreter Development NumPy) # NumPy requires 3.14
message(STATUS "Found Python version ${Python3_VERSION} with NumPy version ${Python3_NumPy_VERSION}")

find_package(pybind11 REQUIRED CONFIG)
message(STATUS "Found pybind11 version ${pybind11_VERSION}")

find_package(xtensor-python REQUIRED CONFIG)
message(STATUS "Found xtensor-python version ${xtensor-python_VERSION}")

# embed the python interpreter into executable
set(SOURCE "${CMAKE_SOURCE_DIR}/src")
set(BINARY "bin")
add_subdirectory("${SOURCE}" "${BINARY}")

# source codes
target_compile_options(
	evolve
	PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>
)
target_include_directories(
	evolve SYSTEM AFTER
	PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>
	PUBLIC ${Python3_INCLUDE_DIRS}
	PUBLIC ${Python3_NumPy_INCLUDE_DIRS}
	PUBLIC ${pybind11_INCLUDE_DIRS}
	PUBLIC ${Eigen3_INCLUDE_DIRS}
	PUBLIC ${xtensor-python_INCLUDE_DIRS}
	PUBLIC ${xtensor_INCLUDE_DIRS}
	PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
	PUBLIC $<INSTALL_INTERFACE:include>
)
target_link_libraries(
	evolve
	PUBLIC xtensor-python
	PUBLIC Python3::NumPy
	PUBLIC pybind11::module
	PUBLIC xtensor
	PUBLIC xtensor::optimize
	PUBLIC Eigen3::Eigen
	PUBLIC $<LINK_ONLY:MKL::MKL>
)
target_compile_options(evolve PRIVATE -march=native)
install(
	TARGETS evolve
	LIBRARY DESTINATION "${BINARY}"
)

# target that moves the python scripts to executable folder
string(
	REPLACE
	";"
	";${SOURCE}/"
	ALL_PY_LIST
	"${SOURCE}/${PY_LIST}"
)
install(
	FILES ${ALL_PY_LIST}
	DESTINATION bin
	PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
)
