## FUNCTIONS TO TRANSLATE KERNELABSTRACTIONS GPU SYMBOLS TO METAL BACKEND HANDLES

function ParallelStencil.ParallelKernel.handle_kernelabstractions_metal()
    return Metal.MetalBackend()
end
