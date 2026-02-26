const ERRMSG_METALEXT_NOT_LOADED = "the Metal extension was not loaded. Make sure to import Metal before ParallelStencil."

# shared.jl

function get_priority_metalstream end
function get_metalstream end
function get_metal_compute_capability end


# select_hardware.jl

handle_kernelabstractions_metal(arg...)  = @NotLoadedError(ERRMSG_METALEXT_NOT_LOADED)


# allocators

zeros_metal(arg...)  = @NotLoadedError(ERRMSG_METALEXT_NOT_LOADED)
ones_metal(arg...)   = @NotLoadedError(ERRMSG_METALEXT_NOT_LOADED)
rand_metal(arg...)   = @NotLoadedError(ERRMSG_METALEXT_NOT_LOADED)
falses_metal(arg...) = @NotLoadedError(ERRMSG_METALEXT_NOT_LOADED)
trues_metal(arg...)  = @NotLoadedError(ERRMSG_METALEXT_NOT_LOADED)
fill_metal(arg...)   = @NotLoadedError(ERRMSG_METALEXT_NOT_LOADED)
fill_metal!(arg...)  = @NotLoadedError(ERRMSG_METALEXT_NOT_LOADED)
