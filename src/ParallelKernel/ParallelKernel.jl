"""
Module ParallelKernel

Enables writing parallel high-performance kernels and whole applications that can be deployed on both GPUs and CPUs.
Single-architecture backends (CUDA, AMDGPU, Metal, Threads, Polyester) remain fixed to their hardware, while abstraction-layer backends such as KernelAbstractions let you switch the runtime target through [`select_hardware`](@ref) and inspect it with [`current_hardware`](@ref) without reparsing code. Detailed workflow guidance is available in the [interactive prototyping runtime selection section](@ref interactive-prototyping-runtime-hardware-selection).

# Usage
    using ParallelStencil.ParallelKernel

# Primary macros
- [`@init_parallel_kernel`](@ref)
- [`@parallel`](@ref)
- [`@hide_communication`](@ref)
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

# Runtime hardware selection
- [`select_hardware`](@ref)
- [`current_hardware`](@ref)

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
    !!! note "Warp-level primitives support"
        Warp-level primitives are only partially supported with the Metal backend.

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
include("parallel.jl")
include("reset_parallel_kernel.jl")
include("select_hardware.jl")

## Alphabetical include of submodules (not extensions)
include("FieldAllocators.jl")

## Exports
export @init_parallel_kernel, @parallel, @hide_communication, @parallel_indices, @parallel_async, @synchronize, @zeros, @ones, @rand, @falses, @trues, @fill, @fill!, @CellType
export select_hardware, current_hardware
export @gridDim, @blockIdx, @blockDim, @threadIdx, @sync_threads, @sharedMem, @pk_show, @pk_println, @∀
export @warpsize, @laneid, @active_mask, @shfl_sync, @shfl_up_sync, @shfl_down_sync, @shfl_xor_sync, @vote_any_sync, @vote_all_sync, @vote_ballot_sync
export PKNumber

end # Module ParallelKernel
