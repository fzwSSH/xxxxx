#!/bin/bash
# Step 2: build Triton-distributed 8260bc3 + the Triton fork f53694a (editable) into $ROOT/.venv.
# ~5-10 min with many cores. The fork's setup.py downloads the NVIDIA toolchain (ptxas 12.8.93 ...).
# usage: ROOT=$HOME/tdist34 bash setup/build34.sh      (SHIM=1 to link the optional glibcxx shim)
set -ex
ROOT=${ROOT:-$HOME/tdist34}
HERE=$(cd $(dirname $0) && pwd)
cd $ROOT
source .venv/bin/activate
TD=$ROOT/Triton-distributed
export LLVM_SYSPATH=$ROOT/llvm/llvm-8957e64a-almalinux-x64
export TRITON_BUILD_PROTON=0
export TRITON_BUILD_LITTLE_KERNEL=0
export USE_TRITON_DISTRIBUTED_AOT=0
export LDFLAGS="-Wl,--version-script=$TD/scripts/triton_hide_libstdcxx.map"
if [ "${SHIM:-0}" = 1 ]; then
  g++ -std=c++17 -O2 -fPIC -c $HERE/glibcxx_shim.cpp -o $ROOT/glibcxx_shim.o
  export TRITON_APPEND_CMAKE_ARGS="-DCMAKE_CXX_STANDARD_LIBRARIES=$ROOT/glibcxx_shim.o"
  # after changing LDFLAGS / cmake args: rm -f $TD/3rdparty/triton/build/cmake*/CMakeCache.txt
fi
export CXXFLAGS="-Wno-attributes"
export MAX_JOBS=${MAX_JOBS:-$(nproc)}
# python-build-standalone's sysconfig asks for clang; force gcc
export CC=gcc CXX=g++ LDSHARED="gcc -shared" LDCXXSHARED="g++ -shared"
cd $TD
uv pip install -e python --no-build-isolation --no-deps -v
python -c "import triton; print('triton', triton.__version__, triton.__file__)"
echo BUILD_DONE
