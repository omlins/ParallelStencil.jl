"""
Module ParallelStencil

Enables domain scientists to write high-level code for parallel high-performance stencil computations that can be deployed on both GPUs and CPUs.

# General overview and examples
https://github.com/omlins/ParallelStencil.jl

# Macros and functions
- [`@init_parallel_stencil`](@ref)
- [`@parallel`](@ref)
- [`@hide_communication`](@ref)
- [`@zeros`](@ref)
- [`@ones`](@ref)
- [`@rand`](@ref)
!!! note "Advanced"
    - [`@parallel_indices`](@ref)
    - [`@parallel_async`](@ref)
    - [`@synchronize`](@ref)

# Submodules
- [`ParallelStencil.FiniteDifferences1D`](@ref)
- [`ParallelStencil.FiniteDifferences2D`](@ref)
- [`ParallelStencil.FiniteDifferences3D`](@ref)

# Modules generated in caller
- [`Data`](@ref)

To see a description of a function, macro or module type `?<functionname>`, `?<macroname>` (including the `@`) or `?<modulename>`, respectively.
"""
module ParallelStencil

## Alphabetical include of submodules, except computation-submodules (below)
include("ParallelKernel/ParallelKernel.jl")
import .ParallelKernel # NOTE: not using .ParallelKernel as for each exported macro except @init_parallel_kernel a corresponding macro is defined in ParallelStencil - using its own doc (modified ParallelKernel doc) and its own check of proper initialization.
import .ParallelKernel: @init_parallel_kernel
using .ParallelKernel.Exceptions

## Include of constant parameters, types and syntax sugar shared in ParallelStencil top module only (must be after include of ParallelKernel.jl as imports from there)
include("shared.jl")

## Alphabetical include of function files
include("init_parallel_stencil.jl")
include("parallel.jl")
include("reset_parallel_stencil.jl")

## Alphabetical include of computation-submodules (must be at end as needs to import from ParallelStencil, .e.g. INDICES).
include("FiniteDifferences.jl")

## Exports (need to be after include of submodules as re-exports from them)
export @init_parallel_stencil, FiniteDifferences1D, FiniteDifferences2D, FiniteDifferences3D
export @parallel, @hide_communication, @parallel_indices, @parallel_async, @synchronize, @zeros, @ones, @rand

end # Module ParallelStencil
