#!/bin/bash
# Build and install MinkowskiEngine v0.5.4 against torch 2.x / CUDA 12.x.
# v0.5.4 predates CUDA 12, so a few source patches are applied first.
# Usage (inside an image with torch, nvcc and g++): build_minkowski_cu12.sh [TORCH_CUDA_ARCH_LIST]
set -ex
ARCH_LIST="${1:-12.0+PTX}"
ME_DIR=/tmp/MinkowskiEngine

rm -rf "$ME_DIR" && git clone -q --depth 1 --branch v0.5.4 https://github.com/NVIDIA/MinkowskiEngine.git "$ME_DIR"
cd "$ME_DIR"

# CUDA 12: thrust headers are no longer pulled in transitively
for f in src/convolution_kernel.cuh src/coordinate_map_gpu.cu src/3rdparty/concurrent_unordered_map.cuh \
         src/spmm.cu src/broadcast_kernel.cu src/pooling_max_kernel.cu; do
  [ -f "$f" ] && sed -i '1i #include <thrust/execution_policy.h>\n#include <thrust/unique.h>\n#include <thrust/remove.h>\n#include <thrust/sort.h>\n#include <thrust/count.h>' "$f"
done

# CUDA 12 removed the legacy cuSPARSE algorithm enums
sed -i 's/CUSPARSE_MV_ALG_DEFAULT/CUSPARSE_SPMV_ALG_DEFAULT/g; s/CUSPARSE_MM_ALG_DEFAULT/CUSPARSE_SPMM_ALG_DEFAULT/g;
        s/CUSPARSE_COOMM_ALG\([123]\)/CUSPARSE_SPMM_COO_ALG\1/g; s/CUSPARSE_CSRMM_ALG1/CUSPARSE_SPMM_CSR_ALG1/g' src/spmm.cu

# CUDA 12's CCCL bundles nvtx3 v1.0, which clashes with ME's older bundled copy.
# ME only uses it for one profiler range (CUDF_FUNC_RANGE), so make that a no-op.
echo '#pragma once' > src/3rdparty/cudf/detail/nvtx/nvtx3.hpp
sed -i 's/^using thread_range = .*$//; s/^#define CUDF_FUNC_RANGE() .*$/#define CUDF_FUNC_RANGE()/' src/3rdparty/cudf/detail/nvtx/ranges.hpp

# torch 2.x: ATen/cuda headers need at::Tensor declared first
sed -i '0,/#include <ATen\/cuda\/CUDAContext.h>/s//#include <ATen\/ATen.h>\n#include <ATen\/cuda\/CUDAContext.h>/' src/spmm.cu src/spmm.cuh

# gcc 11 + CCCL: `shared_ptr = unique_ptr` hits an ambiguous std/cuda::std __to_address
# (found by ADL through thrust::pair). Adopt the raw pointer with its deleter instead.
python3 - <<'PY'
p = 'src/coordinate_map_gpu.cuh'
s = open(p).read()
old = """      m_map = map_type::create(
          compute_hash_table_size(size, m_hashtable_occupancy),
          m_unused_element, m_unused_key, m_hasher, m_equal, m_map_allocator);"""
new = """      {
        auto created = map_type::create(
            compute_hash_table_size(size, m_hashtable_occupancy),
            m_unused_element, m_unused_key, m_hasher, m_equal, m_map_allocator);
        auto deleter = created.get_deleter();
        m_map.reset(created.release(), deleter);
      }"""
assert old in s, 'coordinate_map_gpu.cuh: create() block not found'
open(p, 'w').write(s.replace(old, new))
PY

# Python 3.10 removed the collections.Sequence alias
grep -rl "from collections import Sequence" MinkowskiEngine | xargs sed -i \
    's/^from collections import Sequence, namedtuple$/from collections import namedtuple\nfrom collections.abc import Sequence/; s/^from collections import Sequence$/from collections.abc import Sequence/'

export TORCH_CUDA_ARCH_LIST="$ARCH_LIST" FORCE_CUDA=1 MAX_JOBS="${MAX_JOBS:-$(nproc)}"
python3 setup.py bdist_wheel --force_cuda --blas=openblas
pip install dist/*.whl
cd / && rm -rf "$ME_DIR"
