#!/bin/bash
# Step 1: sources, Python 3.10 venv, LLVM, runtime wheels for the target toolchain:
#   Triton-distributed 8260bc3 (last main commit before the Triton 3.7 switch) with its submodule
#   3rdparty/triton = ByteDance-Seed/triton f53694a (triton.__version__ == "3.4.0", llvm 8957e64a).
# usage: ROOT=$HOME/tdist34 bash setup/setup34.sh      (needs: uv, curl, gcc/g++, ~25 GB disk)
set -ex
ROOT=${ROOT:-$HOME/tdist34}
TD_SHA=8260bc34398c2b8f36dc840fd22f741ca9294584
TR_SHA=f53694a72a1e4f464fa245df2c7305ccda7cb2a9
LLVM=llvm-8957e64a-almalinux-x64
mkdir -p $ROOT && cd $ROOT
uv python install 3.10
[ -d .venv ] || uv venv -p 3.10 .venv
source .venv/bin/activate
uv pip install pybind11 lit wheel "setuptools<80" ninja packaging cmake numpy==2.2.6 RestrictedPython
# sources (tarballs of the exact commits; no git history needed)
if [ ! -d Triton-distributed ]; then
  curl -L --retry 5 -o td.tgz https://codeload.github.com/ByteDance-Seed/Triton-distributed/tar.gz/$TD_SHA
  curl -L --retry 5 -o tr.tgz https://codeload.github.com/ByteDance-Seed/triton/tar.gz/$TR_SHA
  tar xzf td.tgz && tar xzf tr.tgz
  mv Triton-distributed-$TD_SHA Triton-distributed
  rm -rf Triton-distributed/3rdparty/triton
  mv triton-$TR_SHA Triton-distributed/3rdparty/triton
fi
grep -n "__version__" Triton-distributed/3rdparty/triton/python/triton/__init__.py
# LLVM used by the fork (1.2 GB). If this single stream is slow, use setup/pdl.sh instead.
mkdir -p llvm
if [ ! -d llvm/$LLVM ]; then
  curl -L --retry 5 -o llvm/llvm.tgz https://oaitriton.blob.core.windows.net/public/llvm-builds/$LLVM.tar.gz
  tar xzf llvm/llvm.tgz -C llvm
fi
# runtime wheels (versions of the target env; torch 2.8 is the pair Triton-distributed patches for 3.4)
uv pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple
uv pip install cuda-python==12.9.0 cuda-bindings==12.9.9 cuda.core==1.0.1 nvshmem4py-cu12==0.3.0 \
  nvidia-nvshmem-cu12==3.6.5 Cython backports.strenum
# torch pulls the upstream triton wheel; the fork (build34.sh) replaces it
uv pip uninstall triton || true
echo SETUP_DONE
