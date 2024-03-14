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