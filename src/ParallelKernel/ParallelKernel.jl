"""
Module ParallelKernel

Enables writing parallel high-performance kernels and whole applications that can be deployed on both GPUs and CPUs.

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

# Modules generated in caller
- [`Data`](@ref)

To see a description of a macro or module type `?<macroname>` (including the `@`) or `?<modulename>`, respectively.
"""
module ParallelKernel

## Include off exception module
include("Exceptions.jl");
using .Exceptions

## Alphabetical include of submodules.
include(joinpath("EnzymeExt", "AD.jl"));
include("Data.jl");

## Include of constant parameters, types and syntax sugar shared in ParallelKernel module only
include("shared.jl")

## Alphabetical include of function files
include("allocators.jl")
include("hide_communication.jl")
include("init_parallel_kernel.jl")
include("kernel_language.jl")
include("parallel.jl")
include("reset_parallel_kernel.jl")

## Exports
export @init_parallel_kernel, @parallel, @hide_communication, @parallel_indices, @parallel_async, @synchronize, @zeros, @ones, @rand, @falses, @trues, @fill, @fill!, @CellType
export @gridDim, @blockIdx, @blockDim, @threadIdx, @sync_threads, @sharedMem, @pk_show, @pk_println
export PKNumber

end # Module ParallelKernel
