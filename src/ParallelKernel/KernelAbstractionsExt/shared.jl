import ParallelStencil
import KernelAbstractions
using ParallelStencil.ParallelKernel: PKG_KERNELABSTRACTIONS, PKG_CUDA, PKG_AMDGPU, PKG_METAL, PKG_THREADS, PKG_POLYESTER,
    resolve_runtime_backend,
    get_priority_custream, get_custream, get_priority_rocstream, get_rocstream,
    get_priority_metalstream, get_metalstream,
    get_cuda_compute_capability, get_amdgpu_compute_capability, get_metal_compute_capability, get_cpu_compute_capability,
    ERRMSG_UNSUPPORTED_PACKAGE
using ParallelStencil.ParallelKernel.Exceptions: @ArgumentError

const PK = ParallelStencil.ParallelKernel


## FUNCTIONS TO CHECK EXTENSION SUPPORT

PK.is_loaded(::Val{:ParallelStencil_KernelAbstractionsExt}) = true


## FUNCTIONS TO GET OR MANAGE KERNELABSTRACTIONS STREAMS

function dispatch_kastream(kind::Symbol, id::Integer, hardware::Union{Symbol,Nothing})
    target_package, symbol, _ = resolve_runtime_backend(PKG_KERNELABSTRACTIONS, hardware)
    if target_package == PKG_CUDA
        return (kind === :priority) ? get_priority_custream(id) : get_custream(id)
    elseif target_package == PKG_AMDGPU
        return (kind === :priority) ? get_priority_rocstream(id) : get_rocstream(id)
    elseif target_package == PKG_METAL
        return (kind === :priority) ? get_priority_metalstream(id) : get_metalstream(id)
    elseif target_package == PKG_THREADS || target_package == PKG_POLYESTER
        @ArgumentError("KernelAbstractions hardware $(symbol) does not expose GPU streams.")
    else
        @ArgumentError("$(ERRMSG_UNSUPPORTED_PACKAGE) (obtained: $target_package).")
    end
end

function PK.get_priority_kastream(id::Integer; hardware::Union{Symbol,Nothing}=nothing)
    return dispatch_kastream(:priority, id, hardware)
end

function PK.get_kastream(id::Integer; hardware::Union{Symbol,Nothing}=nothing)
    return dispatch_kastream(:regular, id, hardware)
end


## FUNCTIONS TO QUERY DEVICE PROPERTIES

function PK.get_kernelabstractions_compute_capability(default::VersionNumber; hardware::Union{Symbol,Nothing}=nothing)
    target_package, _, _ = resolve_runtime_backend(PKG_KERNELABSTRACTIONS, hardware)
    if target_package == PKG_CUDA
        return get_cuda_compute_capability(default)
    elseif target_package == PKG_AMDGPU
        return get_amdgpu_compute_capability(default)
    elseif target_package == PKG_METAL
        return get_metal_compute_capability(default)
    else
        return get_cpu_compute_capability(default)
    end
end
