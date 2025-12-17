const SELECT_HARDWARE_DOC = let
    base = ParallelKernel.SELECT_HARDWARE_DOC
    base = replace(base, "[`ParallelStencil.select_hardware`](@ref)" => "[`ParallelKernel.select_hardware`](@ref)")
    base = replace(base, "[`ParallelStencil.current_hardware`](@ref)" => "[`ParallelKernel.current_hardware`](@ref)")
    base = replace(base, "@init_parallel_kernel" => "@init_parallel_stencil")
    base = replace(base, "ParallelKernel" => "ParallelStencil")
    base * "\n\nThis wrapper forwards to [`ParallelKernel.select_hardware`](@ref) so stencil-level code can switch hardware without qualifying the kernel namespace."
end
@doc SELECT_HARDWARE_DOC
function select_hardware(hardware::Symbol)
    ParallelKernel.select_hardware(hardware)
end

const CURRENT_HARDWARE_DOC = let
    base = ParallelKernel.CURRENT_HARDWARE_DOC
    base = replace(base, "[`ParallelStencil.select_hardware`](@ref)" => "[`ParallelKernel.select_hardware`](@ref)")
    base = replace(base, "[`ParallelStencil.current_hardware`](@ref)" => "[`ParallelKernel.current_hardware`](@ref)")
    base = replace(base, "@init_parallel_kernel" => "@init_parallel_stencil")
    base = replace(base, "ParallelKernel" => "ParallelStencil")
    base * "\n\nThis wrapper simply returns [`ParallelKernel.current_hardware`](@ref) so you can inspect the runtime selection from the stencil namespace." 
end
@doc CURRENT_HARDWARE_DOC
function current_hardware()
    ParallelKernel.current_hardware()
end
