# easyCuda
easy cuda kernels

# 构建说明

## 开发模式（当前环境构建）
```bash
# 1. 一次性安装构建依赖到当前环境
pip install "scikit-build-core>=0.10" cmake ninja wheel "torch>=2.5.0" --extra-index-url https://download.pytorch.org/whl/cu124

# 2. 之后每次修改代码后，快速重新构建
# Release模式（默认，高性能）
pip install -v . --no-build-isolation

# Debug模式（调试用，包含调试符号）
pip install -v . --no-build-isolation --config-settings=cmake.define.CMAKE_BUILD_TYPE=Debug
```

## 生产模式（隔离环境构建）
```bash
# 默认情况下，pip会创建隔离环境并重新安装依赖，不重用当前环境的包
# Release模式（默认，高性能）
pip install -v . --extra-index-url https://download.pytorch.org/whl/cu124
```