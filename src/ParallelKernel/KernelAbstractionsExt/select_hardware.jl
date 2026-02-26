## FUNCTIONS TO TRANSLATE RUNTIME HARDWARE SYMBOLS TO KERNELABSTRACTIONS HANDLES

function ParallelStencil.ParallelKernel.handle_kernelabstractions(hardware::Symbol)
    if hardware == :cpu
        return KernelAbstractions.CPU()
    elseif hardware == :gpu_cuda
        return ParallelStencil.ParallelKernel.handle_kernelabstractions_cuda()
    elseif hardware == :gpu_amd
        return ParallelStencil.ParallelKernel.handle_kernelabstractions_amd()
    elseif hardware == :gpu_metal
        return ParallelStencil.ParallelKernel.handle_kernelabstractions_metal()
    elseif hardware == :gpu_oneapi
        return ParallelStencil.ParallelKernel.handle_kernelabstractions_oneapi()
    end
    @ArgumentError("unsupported KernelAbstractions hardware symbol (obtained: $hardware).")
end
