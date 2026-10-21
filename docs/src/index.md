```@raw html
---
layout: home

hero:
  name: ParallelStencil.jl
  text: Architecture-agnostic high-performance stencil computations
  tagline: Write high-level code once, run it on GPUs and CPUs, from a single thread to thousands of GPUs.
  actions:
    - theme: brand
      text: Usage 🚀
      link: /usage
    - theme: alt
      text: API reference 📚
      link: /api
    - theme: alt
      text: View on GitHub
      link: https://github.com/omlins/ParallelStencil.jl
  image:
    src: /logo.png
    alt: ParallelStencil.jl

features:
  - icon: ⚡
    title: One macro call
    details: "<code>@parallel</code> parallelizes and optimizes a kernel and launches it; arrays are allocated on the selected hardware."
    link: /usage#usage-parallel
  - icon: 🧮
    title: Math-close notation
    details: Finite differences in 1-D, 2-D and 3-D, readily extensible with custom macros.
    link: /usage#usage-math-close
  - icon: 🖥️
    title: GPUs and CPUs
    details: CUDA, AMDGPU, Metal, Threads and Polyester backends, and KernelAbstractions with runtime hardware selection.
    link: /usage#usage-backends
  - icon: 🌐
    title: Multi-xPU at scale
    details: "Interoperable with ImplicitGlobalGrid.jl and MPI.jl; <code>@hide_communication</code> hides communication behind computation."
    link: /usage#usage-communication
  - icon: 🔧
    title: Low-level kernels
    details: "Index-based <code>@parallel_indices</code> kernels with shared memory and warp-level primitives."
    link: /usage#usage-low-level
  - icon: 🧊
    title: Arrays of small arrays / structs
    details: Logical arrays of small arrays or structs for optimized data access, relying on CellArrays.jl.
    link: /usage#usage-cellarrays
  - icon: ∇
    title: Automatic differentiation
    details: Differentiate architecture-agnostic parallel kernels with Enzyme.jl, using high-level or generic syntax.
    link: /usage#usage-ad
  - icon: 🌋
    title: Miniapps
    details: Concise multi-physics applications, from thermal convection to hydro-mechanical porosity waves.
    link: /miniapps
---
```

## What is ParallelStencil.jl?

ParallelStencil empowers domain scientists to write architecture-agnostic high-level code for parallel high-performance stencil computations on GPUs and CPUs. Performance similar to CUDA C / HIP can be achieved, which is typically a large improvement over the performance reached when using only [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) or [AMDGPU.jl](https://github.com/JuliaGPU/AMDGPU.jl) [GPU Array programming](https://cuda.juliagpu.org/stable/usage/array/). In conjunction with [ImplicitGlobalGrid.jl](https://github.com/eth-cscs/ImplicitGlobalGrid.jl), the same Julia code can run on a single CPU thread or on thousands of GPUs/CPUs. See the [Overview](overview.md) for more details.

## Installation
ParallelStencil can be installed directly with the [Julia package manager](https://docs.julialang.org/en/v1/stdlib/Pkg/) from the [Julia REPL](https://docs.julialang.org/en/v1/stdlib/REPL/):
```julia
import Pkg; Pkg.add("ParallelStencil")
```

## Dependencies
ParallelStencil relies on the Julia packages [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) [Besard2019TPDS, Besard2019AES](@cite), [AMDGPU.jl](https://github.com/JuliaGPU/AMDGPU.jl), [Metal.jl](https://github.com/JuliaGPU/Metal.jl), [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl), [Polyester.jl](https://github.com/JuliaSIMD/Polyester.jl), [MacroTools.jl](https://github.com/FluxML/MacroTools.jl), [CellArrays.jl](https://github.com/omlins/CellArrays.jl), [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl) and [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl). However, thanks to Julia's package extensions, only [CellArrays.jl](https://github.com/omlins/CellArrays.jl), [MacroTools.jl](https://github.com/FluxML/MacroTools.jl) and [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl) are fixed dependencies that are automatically installed with ParallelStencil; the other packages are only required when using the respective backends or features.

## Questions, comments and discussions
To discuss technical issues, please post on Julia Discourse in the [GPU topic](https://discourse.julialang.org/c/domain/gpu) or the [Julia at Scale topic](https://discourse.julialang.org/c/domain/parallel) or in the `#gpu` or `#hpc` channels on the [Julia Slack](https://julialang.org/slack/).
To discuss numerical/domain-science issues, please post on Julia Discourse in the [Numerics topic](https://discourse.julialang.org/c/domain/numerics) or the [Modelling & Simulations topic](https://discourse.julialang.org/c/domain/models) or whichever other topic fits best your issue.

## Contributing
Contributions are welcome, be it new miniapps, math-close notation for other numerical methods (see the tiny [`FiniteDifferences1D`](@ref ParallelStencil.FiniteDifferences1D) submodule as an example), convenience features or core improvements! Please open an issue to discuss your idea beforehand. A pull request should address an issue in its completeness and blend nicely into the existing project (community guidelines such as [ColPrac](https://github.com/SciML/ColPrac) can serve as inspiration).

## Citing
If you use ParallelStencil in your work, please cite [OmlinRaess2024](@cite). The BibTeX entry and further publications are found in the [References](references.md).

## Funding
The development of this package is supported by the [GPU4GEO](https://pasc-ch.org/projects/2021-2024/gpu4geo/index.html) and [∂GPU4GEO](https://pasc-ch.org/projects/2025-2028/gpu4geo-differentiable-multi-physics-solvers-for-extreme-scale-geophysics-simulations-on-gpus/index.html) PASC projects. More information about the GPU4GEO project can be found on the [GPU4GEO website](https://GPU4GEO.org/).
