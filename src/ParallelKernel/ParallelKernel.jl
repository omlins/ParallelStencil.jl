"""
Module ParallelKernel

Enables writing parallel high-performance kernels and whole applications that can be deployed on both GPUs and CPUs.

# Usage
    using ParallelStencil.ParallelKernel

# Macros and functions
- [`@init_parallel_kernel`](@ref)
- [`@parallel`](@ref)
- [`@hide_communication`](@ref)
- [`@zeros`](@ref)
- [`@ones`](@ref)
- [`@rand`](@ref)
!!! note "Advanced"
    - [`@parallel_indices`](@ref)
    - [`@parallel_async`](@ref)
    - [`@synchronize`](@ref)

# Modules generated in caller
- [`Data`](@ref)

To see a description of a function or a macro type `?<functionname>` or `?<macroname>` (including the `@`), respectively.
"""
module ParallelKernel

## Alphabetical include of submodules.
include("Data.jl");
include("Exceptions.jl");
using .Exceptions

## Include of constant parameters, types and syntax sugar shared in ParallelKernel module only
include("shared.jl")

## Alphabetical include of function files
include("allocators.jl")
include("hide_communication.jl")
include("init_parallel_kernel.jl")
include("parallel.jl")
include("reset_parallel_kernel.jl")

## Exports
export @init_parallel_kernel, @parallel, @hide_communication, @parallel_indices, @parallel_async, @synchronize, @zeros, @ones, @rand

end # Module ParallelKernel
