const ERRMSG_CUDAEXT_NOT_LOADED = "the CUDA extension was not loaded. Make sure to import CUDA before ParallelStencil."


# shared.jl

function get_priority_custream end
function get_custream end
function get_cuda_compute_capability end


# select_hardware.jl

handle_kernelabstractions_cuda(arg...)  = @NotLoadedError(ERRMSG_CUDAEXT_NOT_LOADED)


# allocators.jl

zeros_cuda(arg...)  = @NotLoadedError(ERRMSG_CUDAEXT_NOT_LOADED)
ones_cuda(arg...)   = @NotLoadedError(ERRMSG_CUDAEXT_NOT_LOADED)
rand_cuda(arg...)   = @NotLoadedError(ERRMSG_CUDAEXT_NOT_LOADED)
falses_cuda(arg...) = @NotLoadedError(ERRMSG_CUDAEXT_NOT_LOADED)
trues_cuda(arg...)  = @NotLoadedError(ERRMSG_CUDAEXT_NOT_LOADED)
fill_cuda(arg...)   = @NotLoadedError(ERRMSG_CUDAEXT_NOT_LOADED)
fill_cuda!(arg...)  = @NotLoadedError(ERRMSG_CUDAEXT_NOT_LOADED)
