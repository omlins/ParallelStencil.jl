## FUNCTIONS TO TRANSLATE RUNTIME HARDWARE SYMBOLS TO KERNELABSTRACTIONS HANDLES

function ParallelStencil.ParallelKernel.handle_kernelabstractions(hardware::Symbol)
    if hardware == :cpu
        return KernelAbstractions.CPU()
    elseif hardware == :gpu_cuda
        return KernelAbstractions.CUDABackend()
    elseif hardware == :gpu_amd
        return KernelAbstractions.ROCBackend()
    elseif hardware == :gpu_metal
        return KernelAbstractions.MetalBackend()
    elseif hardware == :gpu_oneapi
        return KernelAbstractions.oneAPIBackend()
    end
    @ArgumentError("unsupported KernelAbstractions hardware symbol (obtained: $hardware).")
end
