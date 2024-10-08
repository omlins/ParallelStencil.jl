import ParallelStencil
import ParallelStencil.ParallelKernel: INT_METAL, rand_cpu, fill_cpu, construct_cell, check_datatype, rand_metal, fill_metal
using ParallelStencil.ParallelKernel.Exceptions
using Metal, CellArrays, StaticArrays
import Metal.MTL

## TODO add Metal backend for CellArray
# @define_MetalCellArray

## FUNCTIONS TO CHECK EXTENSIONS SUPPORT
ParallelStencil.ParallelKernel.is_loaded(::Val{:ParallelStencil_MetalExt}) = true

## FUNCTIONS TO GET CREATE AND MANAGE METAL QUEUES
ParallelStencil.ParallelKernel.get_priority_stream(arg...) = get_priority_metalstream(arg...)
ParallelStencil.ParallelKernel.get_metalstream(arg...)     = get_metalstream(arg...)

let 
    global get_priority_metalstream, get_metalstream
    priority_metalqueues = Array{MTL.MTLCommandQueue}(undef, 0)
    metalqueues          = Array{MTL.MTLCommandQueue}(undef, 0)

    function get_priority_metalstream(id::Integer)
        while (id > length(priority_metalqueues)) push!(priority_metalqueues, MTL.MTLCommandQueue(device())) end # No priority setting available in Metal queues.
        return priority_metalqueues[id]
    end

    function get_metalstream(id::Integer)
        while (id > length(metalqueues)) push!(metalqueues, MTL.MTLCommandQueue(device())) end
        return metalqueues[id]
    end
end