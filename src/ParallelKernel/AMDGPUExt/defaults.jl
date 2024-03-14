const ERRMSG_AMDGPUEXT_NOT_LOADED = "the AMDGPU extension was not loaded. Make sure to import AMDGPU before ParallelStencil."


# shared.jl

function get_priority_rocstream end
function get_rocstream end


# allocators.jl

zeros_amdgpu(arg...)  = @NotLoadedError(ERRMSG_AMDGPUEXT_NOT_LOADED)
ones_amdgpu(arg...)   = @NotLoadedError(ERRMSG_AMDGPUEXT_NOT_LOADED)
rand_amdgpu(arg...)   = @NotLoadedError(ERRMSG_AMDGPUEXT_NOT_LOADED)
falses_amdgpu(arg...) = @NotLoadedError(ERRMSG_AMDGPUEXT_NOT_LOADED)
trues_amdgpu(arg...)  = @NotLoadedError(ERRMSG_AMDGPUEXT_NOT_LOADED)
fill_amdgpu(arg...)   = @NotLoadedError(ERRMSG_AMDGPUEXT_NOT_LOADED)
fill_amdgpu!(arg...)  = @NotLoadedError(ERRMSG_AMDGPUEXT_NOT_LOADED)
