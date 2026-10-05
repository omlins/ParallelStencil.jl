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


# warp.jl

const ERRMSG_KERNELABSTRACTIONS_WARP = "warp-level primitives require KernelAbstractions 0.10 or newer (KernelInterface sub-group support)."

warpsize_kernelabstractions(arg...)         = @NotLoadedError(ERRMSG_KERNELABSTRACTIONSEXT_NOT_LOADED)
laneid_kernelabstractions(arg...)           = @NotLoadedError(ERRMSG_KERNELABSTRACTIONSEXT_NOT_LOADED)
active_mask_kernelabstractions(arg...)      = @NotLoadedError(ERRMSG_KERNELABSTRACTIONSEXT_NOT_LOADED)
shfl_sync_kernelabstractions(arg...)        = @NotLoadedError(ERRMSG_KERNELABSTRACTIONSEXT_NOT_LOADED)
shfl_up_sync_kernelabstractions(arg...)     = @NotLoadedError(ERRMSG_KERNELABSTRACTIONSEXT_NOT_LOADED)
shfl_down_sync_kernelabstractions(arg...)   = @NotLoadedError(ERRMSG_KERNELABSTRACTIONSEXT_NOT_LOADED)
shfl_xor_sync_kernelabstractions(arg...)    = @NotLoadedError(ERRMSG_KERNELABSTRACTIONSEXT_NOT_LOADED)
vote_any_sync_kernelabstractions(arg...)    = @NotLoadedError(ERRMSG_KERNELABSTRACTIONSEXT_NOT_LOADED)
vote_all_sync_kernelabstractions(arg...)    = @NotLoadedError(ERRMSG_KERNELABSTRACTIONSEXT_NOT_LOADED)
vote_ballot_sync_kernelabstractions(arg...) = @NotLoadedError(ERRMSG_KERNELABSTRACTIONSEXT_NOT_LOADED)
