#!/usr/bin/env bash
# Builds flip2 for release and packs it as dist/flip2-<build>-linux-x86_64.tar.gz: bin/flip2, built for every GPU a release supports (Turing through
# Blackwell, CMakeLists.txt), the third-party libraries it was built with in lib/ (blosc, and OpenVDB and what it needs when the build found it), which
# flip2 finds there through its RPATH ($ORIGIN/../lib), and a README. The CUDA runtime is linked in; the driver is the machine's own, and has to be one
# for CUDA 13 or newer. Neither the C library nor the C++ runtime is bundled, so the package runs where they're as new as the build machine's or newer:
# build on the oldest system the package should run on (a Rocky Linux 8 container reaches most render farms).
#
#   tools/package-linux.sh [build directory]        (default build-release, beside the repository's own build directory)
#
# FLIP2_BUILD_ID names the build if the source tree isn't a git repository; CUDACXX picks nvcc (default /usr/local/cuda/bin/nvcc).
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
build=${1:-$repo/build-release}
id=$(git -C "$repo" describe --always --dirty 2>/dev/null || echo "${FLIP2_BUILD_ID:-unknown}")
cmake -S "$repo" -B "$build" -DCMAKE_BUILD_TYPE=Release -DFLIP2_CUDA_ARCHITECTURES=release -DFLIP2_BUILD_ID="$id" \
      -DCMAKE_CUDA_COMPILER="${CUDACXX:-/usr/local/cuda/bin/nvcc}" > "$build.configure.log"
cmake --build "$build" --target flip2 -j "$(nproc)"
name=flip2-$id-linux-x86_64
stage=$build/stage/$name
rm -rf "$build/stage"
mkdir -p "$stage/lib"
cmake --install "$build" --prefix "$stage" > /dev/null
# every library it loads that isn't the system's C library, C++ runtime or GPU driver
ldd "$stage/bin/flip2" | awk '/=> \// {print $3}' | while read -r library; do
    case $(basename "$library") in
        libc.so*|libm.so*|libdl.so*|libpthread.so*|librt.so*|libutil.so*|libstdc++.so*|libgcc_s.so*|ld-linux*|libcuda.so*|libnvidia*) ;;
        *) cp -L "$library" "$stage/lib/" ;;
    esac
done
glibc=$(objdump -T "$stage/bin/flip2" "$stage"/lib/*.so* 2>/dev/null | grep -o 'GLIBC_[0-9.]*' | sort -uV | tail -1)
cat > "$stage/README.txt" <<EOF
flip2 $id for Linux x86-64

  bin/flip2 info                  this build, and whether it runs on each GPU here
  bin/flip2 bake scene.json       bake a scene; flip2 with no arguments lists the rest

Built for NVIDIA GPUs from Turing (RTX 20 series) on, with a driver for CUDA 13 or newer. Needs ${glibc:-glibc} or newer.
EOF
mkdir -p "$repo/dist"
tar -C "$build/stage" -czf "$repo/dist/$name.tar.gz" "$name"
echo "$repo/dist/$name.tar.gz"
