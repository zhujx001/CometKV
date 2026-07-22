import os
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

src_dir = "cometkv/src"
cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
os.environ.setdefault("CUDA_HOME", cuda_home)

cuda_compile_args = {
    'cxx': ['-O3', '-std=c++17'],
    'nvcc': ['-O3', '-std=c++17', '--expt-relaxed-constexpr'],
}

ext_modules = [
    CUDAExtension(
        'cometkv.CometKVSignature',
        sources=[f'{src_dir}/cometkv_signature.cu'],
        extra_compile_args=cuda_compile_args,
        extra_link_args=['-lcuda', '-lcudart'],
    ),
    CUDAExtension(
        'cometkv.CometKVGather',
        sources=[f'{src_dir}/cometkv_gather.cu'],
        extra_compile_args=cuda_compile_args,
        extra_link_args=['-lcuda', '-lcudart'],
    ),
    CUDAExtension(
        'cometkv.CometKVGpuGather',
        sources=[f'{src_dir}/cometkv_gpu_gather.cu'],
        extra_compile_args=cuda_compile_args,
        extra_link_args=['-lcuda', '-lcudart'],
    ),
]


setup(
    name='cometkv',
    version='0.2',
    packages=['cometkv'],
    description='CometKV CUDA kernels',
    long_description='CUDA extensions for CometKV signature search and fused KV gather.',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension},
    install_requires=['pybind11', 'torch'],
    python_requires='>=3.10',
)
