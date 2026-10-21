# More examples

The [examples folder](https://github.com/omlins/ParallelStencil.jl/tree/main/examples) contains further examples, each illustrating a particular feature. A simple boolean `USE_GPU` defines whether they run on GPU or CPU. Most of them report the effective memory throughput `T_eff` (see [Performance metric](@ref performance-metric)), and the multi-xPU examples rely on [ImplicitGlobalGrid.jl](https://github.com/eth-cscs/ImplicitGlobalGrid.jl).

## Memory copy

### Memory copy benchmark
A memory copy kernel written with [`@parallel`](@ref) and math-close notation, measuring the effective memory throughput `T_eff` reachable on the hardware (a practical upper bound for stencil computations): [memcopy3D.jl](https://github.com/omlins/ParallelStencil.jl/blob/main/examples/memcopy3D.jl).

::: details Code
```@eval
Main.mdinclude("memcopy3D.jl")
```
:::

### Memory copy benchmark with `@parallel_indices`
The same benchmark written with [`@parallel_indices`](@ref): [memcopy3D\_parindices.jl](https://github.com/omlins/ParallelStencil.jl/blob/main/examples/memcopy3D_parindices.jl).

::: details Code
```@eval
Main.mdinclude("memcopy3D_parindices.jl")
```
:::

### Logical arrays of small arrays / structs
A memory copy of `CellArray`s allocated with the keyword `celldims` and cell types defined with [`@CellType`](@ref), using math-close notation as well as explicit indexing of cells and of cell fields: [memcopyCellArray3D.jl](https://github.com/omlins/ParallelStencil.jl/blob/main/examples/memcopyCellArray3D.jl).

::: details Code
```@eval
Main.mdinclude("memcopyCellArray3D.jl")
```
:::

### Named tuples of arrays
A kernel operating on named tuples of arrays (`Data.NamedArrayTuple`), e.g. on the components of a vector field: [multimemcopyNamedArrayTuple3D.jl](https://github.com/omlins/ParallelStencil.jl/blob/main/examples/multimemcopyNamedArrayTuple3D.jl).

::: details Code
```@eval
Main.mdinclude("multimemcopyNamedArrayTuple3D.jl")
```
:::

## Heat diffusion

### Shared memory
A 2-D heat diffusion solver with a [`@parallel_indices`](@ref) kernel that explicitly uses shared memory ([`@sharedMem`](@ref)): [diffusion2D\_shmem\_novis.jl](https://github.com/omlins/ParallelStencil.jl/blob/main/examples/diffusion2D_shmem_novis.jl).

::: details Code
```@eval
Main.mdinclude("diffusion2D_shmem_novis.jl")
```
:::

### Multi-xPU with performance measurement
The [50-lines multi-xPU example](@ref example-multixpu) with a measurement of `T_eff`: [diffusion3D\_multigpucpu\_novis.jl](https://github.com/omlins/ParallelStencil.jl/blob/main/examples/diffusion3D_multigpucpu_novis.jl).

::: details Code
```@eval
Main.mdinclude("diffusion3D_multigpucpu_novis.jl")
```
:::

### Hiding communication
The communication is hidden behind computation with [`@hide_communication`](@ref): [diffusion3D\_multigpucpu\_hidecomm\_novis.jl](https://github.com/omlins/ParallelStencil.jl/blob/main/examples/diffusion3D_multigpucpu_hidecomm_novis.jl).

::: details Code
```@eval
Main.mdinclude("diffusion3D_multigpucpu_hidecomm_novis.jl")
```
:::

### Hiding communication with `@parallel_indices`
The same with a kernel written with [`@parallel_indices`](@ref): [diffusion3D\_multigpucpu\_hidecomm\_parindices\_novis.jl](https://github.com/omlins/ParallelStencil.jl/blob/main/examples/diffusion3D_multigpucpu_hidecomm_parindices_novis.jl).

::: details Code
```@eval
Main.mdinclude("diffusion3D_multigpucpu_hidecomm_parindices_novis.jl")
```
:::

### Hiding communication with boundary conditions
Boundary conditions are set inside the [`@hide_communication`](@ref) block: [diffusion3Dbc\_multigpucpu\_hidecomm\_novis.jl](https://github.com/omlins/ParallelStencil.jl/blob/main/examples/diffusion3Dbc_multigpucpu_hidecomm_novis.jl).

::: details Code
```@eval
Main.mdinclude("diffusion3Dbc_multigpucpu_hidecomm_novis.jl")
```
:::

### Hiding communication with visualization
The communication is hidden behind computation, and the results are gathered on one process (`gather!` of ImplicitGlobalGrid.jl) to create the movie shown in the [50-lines multi-xPU example](@ref example-multixpu) with [Plots.jl](https://github.com/JuliaPlots/Plots.jl): [diffusion3D\_multigpucpu\_hidecomm.jl](https://github.com/omlins/ParallelStencil.jl/blob/main/examples/diffusion3D_multigpucpu_hidecomm.jl).

::: details Code
```@eval
Main.mdinclude("diffusion3D_multigpucpu_hidecomm.jl")
```
:::

### Number type as parameter
[`@init_parallel_stencil`](@ref) is called without `numbertype`, and the kernels and functions are written generically for any number type (`PSNumber`): [parametrized/diffusion3D\_multigpucpu\_novis.jl](https://github.com/omlins/ParallelStencil.jl/blob/main/examples/parametrized/diffusion3D_multigpucpu_novis.jl).

::: details Code
```@eval
Main.mdinclude("parametrized/diffusion3D_multigpucpu_novis.jl")
```
:::
