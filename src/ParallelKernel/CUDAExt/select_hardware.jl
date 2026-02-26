## FUNCTIONS TO TRANSLATE KERNELABSTRACTIONS GPU SYMBOLS TO CUDA BACKEND HANDLES

function ParallelStencil.ParallelKernel.handle_kernelabstractions_cuda()
    return CUDA.CUDABackend()
end
