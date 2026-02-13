const ERRMSG_KERNELABSTRACTIONSEXT_NOT_LOADED = "the KernelAbstractions extension was not loaded. Make sure to import KernelAbstractions before ParallelStencil."


# shared.jl

function get_kernelabstractions_compute_capability end


# select_hardware.jl

handle_kernelabstractions(arg...)  = @NotLoadedError(ERRMSG_KERNELABSTRACTIONSEXT_NOT_LOADED)


# allocators.jl

zeros_kernelabstractions(arg...)  = @NotLoadedError(ERRMSG_KERNELABSTRACTIONSEXT_NOT_LOADED)
ones_kernelabstractions(arg...)   = @NotLoadedError(ERRMSG_KERNELABSTRACTIONSEXT_NOT_LOADED)
rand_kernelabstractions(arg...)   = @NotLoadedError(ERRMSG_KERNELABSTRACTIONSEXT_NOT_LOADED)
falses_kernelabstractions(arg...) = @NotLoadedError(ERRMSG_KERNELABSTRACTIONSEXT_NOT_LOADED)
trues_kernelabstractions(arg...)  = @NotLoadedError(ERRMSG_KERNELABSTRACTIONSEXT_NOT_LOADED)
fill_kernelabstractions(arg...)   = @NotLoadedError(ERRMSG_KERNELABSTRACTIONSEXT_NOT_LOADED)
fill!_kernelabstractions(arg...)  = @NotLoadedError(ERRMSG_KERNELABSTRACTIONSEXT_NOT_LOADED)
