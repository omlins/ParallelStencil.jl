import ParallelStencil
import ParallelStencil.ParallelKernel: INT_KERNELABSTRACTIONS, rand_cpu, fill_cpu, construct_cell, check_datatype, check_datatype_kernelabstractions, rand_kernelabstractions, fill_kernelabstractions
using ParallelStencil.ParallelKernel.Exceptions
using KernelAbstractions, CellArrays, StaticArrays


## FUNCTIONS TO CHECK EXTENSIONS SUPPORT

ParallelStencil.ParallelKernel.is_loaded(::Val{:ParallelStencil_KernelAbstractionsExt}) = true


## FUNCTIONS TO QUERY DEVICE PROPERTIES

function ParallelStencil.ParallelKernel.get_kernelabstractions_compute_capability(default::VersionNumber)
    compute_capability = default
    return compute_capability
end
