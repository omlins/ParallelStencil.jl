"""
Module ParallelStencil

Enables domain scientists to write high-level code for parallel high-performance stencil computations that can be deployed on both GPUs and CPUs.

# General overview and examples
https://github.com/omlins/ParallelStencil.jl

# Primary macros
- [`@init_parallel_stencil`](@ref)
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
- [`@ps_show`](@ref)
- [`@ps_println`](@ref)
!!! note "Advanced"
    - [`@gridDim`](@ref)
    - [`@blockIdx`](@ref)
    - [`@blockDim`](@ref)
    - [`@threadIdx`](@ref)
    - [`@sync_threads`](@ref)
    - [`@sharedMem`](@ref)

# Submodules
- [`ParallelStencil.AD`](@ref)
- [`ParallelStencil.FieldAllocators`](@ref)
- [`ParallelStencil.FiniteDifferences1D`](@ref)
- [`ParallelStencil.FiniteDifferences2D`](@ref)
- [`ParallelStencil.FiniteDifferences3D`](@ref)

# Modules generated in caller
- [`Data`](@ref)

!! note "Activation of GPU support"
    The support for GPU (CUDA, AMDGPU or Metal) is provided with extensions and requires therefore an explicit installation of the corresponding packages (CUDA.jl, AMDGPU.jl or Metal.jl). Note that it is not required to import explicitly the corresponding module (CUDA, AMDGPU or Metal); this is automatically done by [`@init_parallel_stencil`](@ref).

To see a description of a macro or module type `?<macroname>` (including the `@`) or `?<modulename>`, respectively.
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
include("allocators.jl")
include("hide_communication.jl")
include("kernel_language.jl")
include("init_parallel_stencil.jl")
include("memopt.jl")
include("parallel.jl")
include("reset_parallel_stencil.jl")

## Alphabetical include of allocation/computation-submodules (must be at end as needs to import from ParallelStencil, .e.g. INDICES).
include("AD.jl")
include("FieldAllocators.jl")
include("FiniteDifferences.jl")

## Exports (need to be after include of submodules as re-exports from them)
export @init_parallel_stencil, FiniteDifferences1D, FiniteDifferences2D, FiniteDifferences3D, AD
export @parallel, @hide_communication, @parallel_indices, @parallel_async, @synchronize, @zeros, @ones, @rand, @falses, @trues, @fill, @fill!, @CellType
export @gridDim, @blockIdx, @blockDim, @threadIdx, @sync_threads, @sharedMem, @ps_show, @ps_println, @∀
export @warpsize, @laneid, @active_mask, @shfl_sync, @shfl_up_sync, @shfl_down_sync, @shfl_xor_sync, @vote_any_sync, @vote_all_sync, @vote_ballot_sync
export PSNumber

end # Module ParallelStencil
