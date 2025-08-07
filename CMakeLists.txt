cmake_minimum_required(VERSION 3.22)
project(easyCuda LANGUAGES CXX CUDA)

# ==============================================================================
# 编译命令数据库配置
# ==============================================================================
# 生成 compile_commands.json 文件，用于 IDE 和工具支持
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 添加自定义目标，在构建后复制 compile_commands.json 到项目根目录
add_custom_target(copy_compile_commands ALL
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_BINARY_DIR}/compile_commands.json
        ${CMAKE_SOURCE_DIR}/compile_commands.json
    DEPENDS _C
    COMMENT "Copying compile_commands.json to project root"
)

# ==============================================================================
# 构建类型配置
# ==============================================================================
# 设置默认构建类型为Release（如果没有指定）
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 输出当前构建类型
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)

# 定义CUDA编译选项
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(EASYCUDA_CUDA_FLAGS
        "-g" "-G" "-O0" "-DDEBUG"
        "-Xcompiler=-fPIC"
        "-std=c++17"
        "--expt-relaxed-constexpr"
        "--expt-extended-lambda"
        "--threads=8"
    )
    message(STATUS "Debug mode: enabling debug symbols and disabling optimizations")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(EASYCUDA_CUDA_FLAGS
        "-O3" "-DNDEBUG"
        "-Xcompiler=-fPIC"
        "-std=c++17"
        "-DCUTE_USE_PACKED_TUPLE=1"
        "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1"
        "-DCUTLASS_VERSIONS_GENERATED"
        "-DCUTLASS_TEST_LEVEL=0"
        "-DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1"
        "-DCUTLASS_DEBUG_TRACE_LEVEL=0"
        "--expt-relaxed-constexpr"
        "--expt-extended-lambda"
        "--threads=32"
        "-Xcompiler=-Wconversion"
        "-Xcompiler=-fno-strict-aliasing"
    )
    message(STATUS "Release mode: enabling optimizations and Cutlass support")
endif()


if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 90)
endif()

# ==============================================================================
# 依赖包查找
# ==============================================================================
# find_package(): 查找并配置外部依赖包
# REQUIRED: 表示该包是必需的，如果找不到会终止CMake配置

# 查找PyTorch库，PyTorch提供深度学习框架和CUDA张量操作
find_package(Torch REQUIRED)

# 查找Python开发环境，包含Python头文件和库
# COMPONENTS Development: 指定需要Python的开发组件（头文件、库文件）
find_package(Python REQUIRED COMPONENTS Development)

# 查找CUDA工具包，提供CUDA运行时和驱动程序库
find_package(CUDAToolkit REQUIRED)

# ==============================================================================
# 源文件收集
# ==============================================================================
# file(GLOB_RECURSE): 递归搜索匹配模式的文件
# CONFIGURE_DEPENDS: 当文件列表变化时自动重新配置CMake

# 收集Python绑定相关的C++源文件（通常包含pybind11绑定代码）
file(GLOB_RECURSE SRC_CPP   CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/binds/*.cpp")

# 收集所有CUDA核函数源文件（包含GPU计算代码）
file(GLOB_RECURSE SRC_CUDA  CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/csrc/*.cu")

# ==============================================================================
# Python扩展模块编译
# ==============================================================================
# 检查目标是否已存在，避免重复定义
if (NOT TARGET _C)
    # add_library(): 创建一个库目标
    # _C: 库名称，这是Python扩展模块的名称
    # SHARED: 创建动态链接库(.so文件)，Python可以动态加载
    add_library(_C SHARED ${SRC_CPP} ${SRC_CUDA})

    # target_compile_features(): 设置编译特性要求
    # cxx_std_17: 要求使用C++17标准，提供现代C++语法支持
    # PRIVATE: 该要求仅应用于当前目标，不传播给依赖此目标的其他目标
    target_compile_features(_C PRIVATE cxx_std_17)
    
    # target_include_directories(): 添加头文件搜索路径
    # PRIVATE: 头文件路径仅对当前目标可见
    target_include_directories(_C PRIVATE
            ${TORCH_INCLUDE_DIRS}          # PyTorch头文件路径
            ${CUDAToolkit_INCLUDE_DIRS}    # CUDA工具包头文件路径
            ${PROJECT_SOURCE_DIR}/include  # 项目公共头文件目录
            ${PROJECT_SOURCE_DIR}/third-party/cutlass/include # cutlass && cute头文件目录
            ${PROJECT_SOURCE_DIR}/csrc     # CUDA源码头文件目录
            ${PROJECT_SOURCE_DIR}/binds    # Python绑定头文件目录
    )
    
    # target_link_libraries(): 链接外部库
    # PRIVATE: 库依赖仅对当前目标可见，不传播
    target_link_libraries(_C
            PRIVATE
            ${TORCH_LIBRARIES}      # PyTorch库文件
            Python::Python         # Python开发库
            CUDA::cudart           # CUDA运行时库
            CUDA::cuda_driver)     # CUDA驱动程序库

    # target_compile_options(): 设置编译选项
    # 只对CUDA文件应用CUDA特定的编译选项
    target_compile_options(_C PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${EASYCUDA_CUDA_FLAGS}>)

    # set_target_properties(): 设置目标属性
    # PREFIX "": 移除库文件名前缀（通常是"lib"）
    # OUTPUT_NAME "_C": 设置输出文件名为_C.so
    # 注意：这里没有设置LIBRARY_OUTPUT_DIRECTORY，使用CMake默认行为
    # 默认输出到构建目录（${CMAKE_CURRENT_BINARY_DIR}），通常是build/
    # BUILD_RPATH/INSTALL_RPATH: 设置运行时库搜索路径，确保能找到PyTorch库
    set_target_properties(_C PROPERTIES
            PREFIX ""                                    # 去掉"lib"前缀
            OUTPUT_NAME "_C"                            # 输出文件名为_C.so
            LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/easyCuda # 输出_C.so到easyCuda目录
            BUILD_RPATH "\$ORIGIN;\$ORIGIN/../torch/lib"     # 编译时的库搜索路径
            INSTALL_RPATH "\$ORIGIN;\$ORIGIN/../torch/lib"   # 安装后的库搜索路径
    )
endif()

# ==============================================================================
# 安装配置
# ==============================================================================
# include(): 包含CMake模块，提供标准安装路径变量
include(GNUInstallDirs)

# message(): 输出状态信息到控制台
# STATUS: 信息级别的消息
# Python_SITEARCH: platlib 平台库, 可通过"""python -c "from sysconfig import get_path; print(get_path('platlib'))"""" 获取
# Python_SITELIB: purelib 纯Python库, 可通过"""python -c "from sysconfig import get_path; print(get_path('purelib'))"""" 获取
# platlib 与 purelib 路径可能相同或者不同
message(STATUS "Installing to ARCH = ${Python_SITEARCH}")
message(STATUS "Installing to PURE = ${Python_SITELIB}")

# 第1步：先安装Python源码包
# install(): 安装目录
# DIRECTORY: 指定要安装的目录  
# DESTINATION: 安装目标路径
# 排除Python缓存目录
install(DIRECTORY ${PROJECT_SOURCE_DIR}/easyCuda
        DESTINATION ${Python_SITEARCH}
        PATTERN "__pycache__" EXCLUDE)

# 第2步：再安装编译好的_C.so（不会覆盖目录）
# install(): 定义安装规则
# TARGETS: 安装指定的目标
# LIBRARY DESTINATION: 动态库的安装目标路径
install(TARGETS _C
        LIBRARY DESTINATION ${Python_SITEARCH}/easyCuda)
