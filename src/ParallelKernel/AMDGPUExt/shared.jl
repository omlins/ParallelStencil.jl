import ParallelStencil
import ParallelStencil.ParallelKernel: INT_AMDGPU, rand_cpu, fill_cpu, construct_cell, check_datatype, rand_amdgpu, fill_amdgpu
using ParallelStencil.ParallelKernel.Exceptions
using AMDGPU, CellArrays, StaticArrays
@define_ROCCellArray


## FUNCTIONS TO CHECK EXTENSIONS SUPPORT

ParallelStencil.ParallelKernel.is_loaded(::Val{:ParallelStencil_AMDGPUExt}) = true


## FUNCTIONS TO GET CREATE AND MANAGE AMDGPU QUEUES AND "ROCSTREAMS"

ParallelStencil.ParallelKernel.get_priority_rocstream(arg...) = get_priority_rocstream(arg...)
ParallelStencil.ParallelKernel.get_rocstream(arg...)          = get_rocstream(arg...)
let
    global get_priority_rocstream, get_rocstream
    priority_rocstreams = Array{AMDGPU.HIPStream}(undef, 0)
    rocstreams          = Array{AMDGPU.HIPStream}(undef, 0)

    function get_priority_rocstream(id::Integer)
        while (id > length(priority_rocstreams)) push!(priority_rocstreams, AMDGPU.HIPStream(:high)) end
        return priority_rocstreams[id]
    end

    function get_rocstream(id::Integer)
        while (id > length(rocstreams)) push!(rocstreams, AMDGPU.HIPStream(:low)) end
        return rocstreams[id]
    end
end


## FUNCTIONS TO SUSPEND THE IMPLICIT SYNCHRONIZATION OF ARRAYS (see hide_communication.jl)

ParallelStencil.ParallelKernel.collect_gpu_arrays!(arrays, x::ROCArray) = push!(arrays, x)

function ParallelStencil.ParallelKernel.implicit_sync_off!(A::ROCArray)
    if !isdefined(AMDGPU, :enable_synchronization!) return nothing end # NOTE: AMDGPU.jl versions without it synchronize implicitly in any case.
    state = isdefined(AMDGPU, :synchronization_enabled) ? AMDGPU.synchronization_enabled(A) : true # NOTE: the query and synchronize(::ROCArray) were added to AMDGPU.jl together with enable_synchronization!; each is only used if present, so that PS does not fail if they change.
    if hasmethod(AMDGPU.synchronize, Tuple{typeof(A)}) AMDGPU.synchronize(A) end # NOTE: wait for the work pending on A on the stream that used it last, as the implicit synchronization would.
    AMDGPU.enable_synchronization!(A, false)
    return state
end

ParallelStencil.ParallelKernel.implicit_sync_set!(A::ROCArray, state::Bool) = (AMDGPU.enable_synchronization!(A, state); return)


## FUNCTIONS TO QUERY DEVICE PROPERTIES

function ParallelStencil.ParallelKernel.get_amdgpu_compute_capability(default::VersionNumber)
    compute_capability = default
    #TODO: implement and convert to something comparable to CUDA compute capability.
    return compute_capability
end
