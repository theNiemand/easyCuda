cmake_minimum_required(VERSION 3.22)
project(easyCuda LANGUAGES CXX CUDA)

if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

# 找 PyTorch & Python
find_package(Torch REQUIRED)
find_package(Python REQUIRED COMPONENTS Development)
find_package(CUDAToolkit REQUIRED)

# 收集 .cpp/.cu
file(GLOB_RECURSE SRC_CPP   CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/binds/*.cpp")
file(GLOB_RECURSE SRC_CUDA  CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/csrc/*.cu")

# 编译生成 Python 扩展， _C.so
if (NOT TARGET _C)
    add_library(_C SHARED ${SRC_CPP} ${SRC_CUDA})

    # C++17 更方便调度宏
    target_compile_features(_C PRIVATE cxx_std_17)
    target_include_directories(_C PRIVATE
            ${TORCH_INCLUDE_DIRS}
            ${CUDAToolkit_INCLUDE_DIRS}
            ${PROJECT_SOURCE_DIR}/include
            ${PROJECT_SOURCE_DIR}/csrc
            ${PROJECT_SOURCE_DIR}/binds
    )
    target_link_libraries(_C
            PRIVATE
            ${TORCH_LIBRARIES}
            Python::Python
            CUDA::cudart
            CUDA::cuda_driver)


    # 输出文件名 _C.so，无前缀
    set_target_properties(_C PROPERTIES
            PREFIX ""
            OUTPUT_NAME "_C"
            LIBRARY_OUTPUT_DIRECTORY
            ${PROJECT_SOURCE_DIR}/easyCuda
            BUILD_RPATH "\$ORIGIN;\$ORIGIN/../torch/lib"
            INSTALL_RPATH "\$ORIGIN;\$ORIGIN/../torch/lib"
    )
endif()
# 安装：把 _C.so、Python 包和 csrc 一起拷到 site-packages
include(GNUInstallDirs)

# 1) 计算 Python site-packages 路径
message(STATUS "Installing to ARCH = ${Python_SITEARCH}")
message(STATUS "Installing to PURE = ${Python_SITELIB}")

# 2) 安装编译好的 _C.so 到 easyCuda 目录
install(TARGETS _C
        LIBRARY DESTINATION ${Python_SITEARCH}/easyCuda)

# 3) 安装 Python 源码包
install(DIRECTORY ${PROJECT_SOURCE_DIR}/easyCuda
        DESTINATION ${Python_SITELIB})
