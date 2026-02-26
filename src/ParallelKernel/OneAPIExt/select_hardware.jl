import oneAPI


## FUNCTIONS TO TRANSLATE KERNELABSTRACTIONS GPU SYMBOLS TO ONEAPI BACKEND HANDLES

function ParallelStencil.ParallelKernel.handle_kernelabstractions_oneapi()
    return oneAPI.oneAPIBackend()
end
