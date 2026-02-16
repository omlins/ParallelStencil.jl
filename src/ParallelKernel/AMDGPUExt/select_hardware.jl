## FUNCTIONS TO TRANSLATE KERNELABSTRACTIONS GPU SYMBOLS TO AMDGPU BACKEND HANDLES

function ParallelStencil.ParallelKernel.handle_kernelabstractions_amd()
    return AMDGPU.ROCBackend()
end
