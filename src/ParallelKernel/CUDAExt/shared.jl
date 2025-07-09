import ParallelStencil
import ParallelStencil.ParallelKernel: INT_CUDA, rand_cpu, fill_cpu, construct_cell, check_datatype, rand_cuda, fill_cuda
using ParallelStencil.ParallelKernel.Exceptions
using CUDA, CellArrays, StaticArrays
@define_CuCellArray


## FUNCTIONS TO CHECK EXTENSIONS SUPPORT

ParallelStencil.ParallelKernel.is_loaded(::Val{:ParallelStencil_CUDAExt}) = true


## FUNCTIONS TO GET CREATE AND MANAGE CUDA STREAMS

ParallelStencil.ParallelKernel.get_priority_custream(arg...) = get_priority_custream(arg...)
ParallelStencil.ParallelKernel.get_custream(arg...)          = get_custream(arg...)
let
    global get_priority_custream, get_custream
    priority_custreams = Array{CuStream}(undef, 0)
    custreams          = Array{CuStream}(undef, 0)

    function get_priority_custream(id::Integer)
        while (id > length(priority_custreams)) push!(priority_custreams, CuStream(; flags=CUDA.STREAM_NON_BLOCKING, priority=CUDA.priority_range()[end])) end # CUDA.priority_range()[end] is max priority. # NOTE: priority_range cannot be called outside the function as only at runtime sure that CUDA is functional.
        return priority_custreams[id]
    end

    function get_custream(id::Integer)
        while (id > length(custreams)) push!(custreams, CuStream(; flags=CUDA.STREAM_NON_BLOCKING, priority=CUDA.priority_range()[1])) end # CUDA.priority_range()[1] is min priority. # NOTE: priority_range cannot be called outside the function as only at runtime sure that CUDA is functional.
        return custreams[id]
    end
end


## FUNCTIONS TO QUERY DEVICE PROPERTIES

function ParallelStencil.ParallelKernel.get_cuda_compute_capability(default::VersionNumber)
    compute_capability = default
    if haskey(ENV, "PS_CUDA_COMPUTE_CAPABILITY")
        compute_capability = parse(VersionNumber, ENV["PS_CUDA_COMPUTE_CAPABILITY"])
    else
        try
            dev = CUDA.device()
            compute_capability = CUDA.capability(dev)
        catch e
            @warn "Could not determine CUDA compute capability: assuming a recent architecture. Set the environment variable PS_CUDA_COMPUTE_CAPABILITY to a specific value if desired."
        end
    end
    return compute_capability
end
