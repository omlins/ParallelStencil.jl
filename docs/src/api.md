```@meta
CurrentModule = ParallelStencil
```

# API reference

This is the official API reference of ParallelStencil. Note that it can also be queried interactively from the [Julia REPL](https://docs.julialang.org/en/v1/stdlib/REPL/) using the [help mode](https://docs.julialang.org/en/v1/stdlib/REPL/#Help-mode):
```julia-repl
julia> using ParallelStencil
julia>?
help?> ParallelStencil
```

## Module
```@docs
ParallelStencil
```

## Initialization
```@docs
@init_parallel_stencil
```

## Parallel kernels
```@docs
@parallel
@parallel_indices
@parallel_async
@synchronize
```

## Allocation
```@docs
@zeros
@ones
@rand
@falses
@trues
@fill
@fill!
@CellType
```

## Runtime hardware selection
```@docs
@select_hardware
@current_hardware
select_hardware
current_hardware
```

## Hiding communication
```@docs
@hide_communication
@overlap
@get_stream
@get_priority_stream
```

## Kernel language
The following macros are available in [`@parallel_indices`](@ref) kernels.
```@docs
@ps_show
@ps_println
@∀
@gridDim
@blockIdx
@blockDim
@threadIdx
@sync_threads
@sharedMem
```

### Warp-level primitives
```@docs
@warpsize
@laneid
@active_mask
@shfl_sync
@shfl_up_sync
@shfl_down_sync
@shfl_xor_sync
@vote_any_sync
@vote_all_sync
@vote_ballot_sync
```

## [Module `Data`](@id data-module)
The module `Data` is generated in the caller module by [`@init_parallel_stencil`](@ref) (if a `numbertype` is given; type `?Data` after the call to see its description in the [Julia REPL](https://docs.julialang.org/en/v1/stdlib/REPL/)):
```@eval
using ParallelStencil, Markdown
datadoc = replace(ParallelStencil.ParallelKernel.DATA_DOC, "ParallelKernel" => "ParallelStencil", "@init_parallel_kernel" => "@init_parallel_stencil")
Markdown.parse(replace(datadoc, r"\[(`[^`]*`)\]\(@ref\)" => s"\1"))  # Cross-references are not resolved in the output of `@eval` blocks.
```

## Finite differences
### 1-D
```@autodocs
Modules = [ParallelStencil.FiniteDifferences1D]
Order   = [:module, :macro]
```

### 2-D
```@autodocs
Modules = [ParallelStencil.FiniteDifferences2D]
Order   = [:module, :macro]
```

### 3-D
```@autodocs
Modules = [ParallelStencil.FiniteDifferences3D]
Order   = [:module, :macro]
```

## Field allocators
```@autodocs
Modules = [ParallelStencil.FieldAllocators]
Order   = [:module, :macro]
```

## Automatic differentiation
```@docs
ParallelStencil.AD
```
