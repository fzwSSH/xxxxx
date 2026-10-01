#!/bin/bash
# Optional: parallel ranged download of the LLVM 8957e64a tarball (32 parts), for slow links.
# usage: ROOT=$HOME/tdist34 bash setup/pdl.sh
ROOT=${ROOT:-$HOME/tdist34}
U=https://oaitriton.blob.core.windows.net/public/llvm-builds/llvm-8957e64a-almalinux-x64.tar.gz
S=1215526612; N=32; P=$(( (S + N - 1) / N ))
mkdir -p $ROOT/llvm && cd $ROOT/llvm; rm -f part.* llvm.tgz
for i in $(seq 0 $((N-1))); do
  a=$((i*P)); b=$((a+P-1)); [ $b -ge $S ] && b=$((S-1))
  ( for t in 1 2 3 4 5; do curl -sL --retry 5 -r $a-$b -o part.$(printf %02d $i) $U && [ $(stat -c %s part.$(printf %02d $i)) -eq $((b-a+1)) ] && break; done ) &
done
wait
cat part.* > llvm.tgz && rm -f part.* && ls -la llvm.tgz && tar xzf llvm.tgz && echo LLVM_OK
