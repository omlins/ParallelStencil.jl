"""
    @reset_parallel_stencil()

Reset the ParallelStencil module.

See also: [`init_parallel_stencil`](@ref)
"""
macro reset_parallel_stencil() esc(reset_parallel_stencil(__module__)) end

function reset_parallel_stencil(caller::Module)
    ParallelKernel.reset_parallel_kernel(caller)
    set_initialized(caller, false)
    set_package(caller, PKG_NONE)
    set_numbertype(caller, NUMBERTYPE_NONE)
    set_ndims(caller, NDIMS_NONE)
    return nothing
end
