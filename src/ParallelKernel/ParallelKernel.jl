"""
Module ParallelKernel

Enables writing parallel high-performance kernels and whole applications that can be deployed on both GPUs and CPUs.

# Usage
    using ParallelStencil.ParallelKernel

# Primary macros
- [`@init_parallel_kernel`](@ref)
- [`@parallel`](@ref)
- [`@hide_communication`](@ref)
- [`@overlap`](@ref)
- [`@zeros`](@ref)
- [`@ones`](@ref)
- [`@rand`](@ref)
- [`@falses`](@ref)
- [`@trues`](@ref)
- [`@fill`](@ref)
- [`@fill!`](@ref)
!!! note "Advanced"
    - [`@parallel_indices`](@ref)
    - [`@parallel_async`](@ref)
    - [`@synchronize`](@ref)
    - [`@get_stream`](@ref)
    - [`@get_priority_stream`](@ref)

# Macros available for [`@parallel_indices`](@ref) kernels
- [`@pk_show`](@ref)
- [`@pk_println`](@ref)
!!! note "Advanced"
    - [`@gridDim`](@ref)
    - [`@blockIdx`](@ref)
    - [`@blockDim`](@ref)
    - [`@threadIdx`](@ref)
    - [`@sync_threads`](@ref)
    - [`@sharedMem`](@ref)
!!! note "Warp-level primitives"
    - [`@warpsize`](@ref)
    - [`@laneid`](@ref)
    - [`@active_mask`](@ref)
    - [`@shfl_sync`](@ref)
    - [`@shfl_up_sync`](@ref)
    - [`@shfl_down_sync`](@ref)
    - [`@shfl_xor_sync`](@ref)
    - [`@vote_any_sync`](@ref)
    - [`@vote_all_sync`](@ref)
    - [`@vote_ballot_sync`](@ref)

# Submodules
- [`ParallelKernel.AD`](@ref)
- [`ParallelKernel.FieldAllocators`](@ref)

# Modules generated in caller
- [`Data`](@ref)

To see a description of a macro or module type `?<macroname>` (including the `@`) or `?<modulename>`, respectively.
"""
module ParallelKernel

## Include of exception module
include("Exceptions.jl")
using .Exceptions

## Alphabetical include of submodules for extensions
include(joinpath("EnzymeExt", "AD.jl"))

## Alphabetical include of defaults for extensions
include(joinpath("AMDGPUExt", "defaults.jl"))
include(joinpath("CUDAExt", "defaults.jl"))
include(joinpath("MetalExt", "defaults.jl"))

## Include of constant parameters, types and syntax sugar shared in ParallelKernel module only
include("shared.jl")

## Alphabetical include of function files
include("allocators.jl")
include("Data.jl")
include("hide_communication.jl")
include("init_parallel_kernel.jl")
include("kernel_language.jl")
include("overlap.jl")
include("parallel.jl")
include("reset_parallel_kernel.jl")

## Alphabetical include of submodules (not extensions)
include("FieldAllocators.jl")

## Exports
export @init_parallel_kernel, @parallel, @hide_communication, @overlap, @get_priority_stream, @get_stream, @parallel_indices, @parallel_async, @synchronize, @zeros, @ones, @rand, @falses, @trues, @fill, @fill!, @CellType
export @gridDim, @blockIdx, @blockDim, @threadIdx, @sync_threads, @sharedMem, @pk_show, @pk_println, @∀
export @warpsize, @laneid, @active_mask, @shfl_sync, @shfl_up_sync, @shfl_down_sync, @shfl_xor_sync, @vote_any_sync, @vote_all_sync, @vote_ballot_sync
export PKNumber

end # Module ParallelKernel
