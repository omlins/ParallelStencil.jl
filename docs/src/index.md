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
---
```

## What is ParallelStencil.jl?

ParallelStencil empowers domain scientists to write architecture-agnostic high-level code for parallel high-performance stencil computations on GPUs and CPUs. Performance similar to CUDA C / HIP can be achieved, which is typically a large improvement over the performance reached when using only [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) or [AMDGPU.jl](https://github.com/JuliaGPU/AMDGPU.jl) [GPU Array programming](https://cuda.juliagpu.org/stable/usage/array/). In conjunction with [ImplicitGlobalGrid.jl](https://github.com/eth-cscs/ImplicitGlobalGrid.jl), the same Julia code can run on a single CPU thread or on thousands of GPUs/CPUs. See the [Overview](overview.md) for more details.

- **One macro call** parallelizes and optimizes a kernel and launches it: [`@parallel`](@ref) (see [Usage](usage.md)).
- **Math-close notation** for finite differences in 1-D, 2-D and 3-D, e.g. with [`FiniteDifferences3D`](@ref ParallelStencil.FiniteDifferences3D).
- **GPU and CPU backends**: [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl), [AMDGPU.jl](https://github.com/JuliaGPU/AMDGPU.jl), [Metal.jl](https://github.com/JuliaGPU/Metal.jl), [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) (with runtime hardware selection), [Base.Threads](https://docs.julialang.org/en/v1/base/multi-threading/) and [Polyester.jl](https://github.com/JuliaSIMD/Polyester.jl).
- **Hiding communication** behind computation with one macro call: [`@hide_communication`](@ref).
- **Low-level kernel programming** with architecture-agnostic constructs (shared memory, warp-level primitives) in [`@parallel_indices`](@ref) kernels.
- **Logical arrays of small arrays / structs** relying on [CellArrays.jl](https://github.com/omlins/CellArrays.jl).
- **Automatic differentiation** of architecture-agnostic parallel kernels relying on [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl).

## Installation
ParallelStencil can be installed directly with the [Julia package manager](https://docs.julialang.org/en/v1/stdlib/Pkg/) from the [Julia REPL](https://docs.julialang.org/en/v1/stdlib/REPL/):
```julia
import Pkg; Pkg.add("ParallelStencil")
```

## Dependencies
ParallelStencil relies on the Julia packages [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) \[[5](https://doi.org/10.1109/TPDS.2018.2872064), [6](https://doi.org/10.1016/j.advengsoft.2019.02.002)\], [AMDGPU.jl](https://github.com/JuliaGPU/AMDGPU.jl), [Metal.jl](https://github.com/JuliaGPU/Metal.jl), [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl), [Polyester.jl](https://github.com/JuliaSIMD/Polyester.jl), [MacroTools.jl](https://github.com/FluxML/MacroTools.jl), [CellArrays.jl](https://github.com/omlins/CellArrays.jl), [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl) and [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl). However, thanks to Julia's package extensions, only [CellArrays.jl](https://github.com/omlins/CellArrays.jl), [MacroTools.jl](https://github.com/FluxML/MacroTools.jl) and [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl) are fixed dependencies that are automatically installed with ParallelStencil; the other packages are only required when using the respective backends or features.

## Questions, comments and discussions
To discuss technical issues, please post on Julia Discourse in the [GPU topic](https://discourse.julialang.org/c/domain/gpu) or the [Julia at Scale topic](https://discourse.julialang.org/c/domain/parallel) or in the `#gpu` or `#hpc` channels on the [Julia Slack](https://julialang.org/slack/).
To discuss numerical/domain-science issues, please post on Julia Discourse in the [Numerics topic](https://discourse.julialang.org/c/domain/numerics) or the [Modelling & Simulations topic](https://discourse.julialang.org/c/domain/models) or whichever other topic fits best your issue.

## Contributing
This project welcomes your contribution! Have you developed an application with ParallelStencil that could be featured as a [miniapp](miniapps.md)? Please contribute it to share it with the world! Would you like to use other methods than finite differences with math-close notation in ParallelStencil kernels? Then check out the tiny [`FiniteDifferences1D`](@ref ParallelStencil.FiniteDifferences1D) submodule as an example for enabling math-close notation for a method and contribute your own submodule! Are you missing a great feature in the core of ParallelStencil? Maybe you can contribute yourself!

Please open an issue to discuss your idea for a contribution beforehand. Furthermore, note that a pull request should always address a significant issue in its completeness. Moreover, pull requests should blend nicely into the existing project; common sense is the primary guide in this regard (community guideline documents, e.g. [ColPrac](https://github.com/SciML/ColPrac), can be consulted in addition for inspiration). We are looking forward to your contribution!

## References
\[1\] [Omlin, S., Räss, L., Kwasniewski, G., Malvoisin, B., & Podladchikov, Y. Y. (2020). Solving Nonlinear Multi-Physics on GPU Supercomputers with Julia. JuliaCon Conference, virtual.](https://www.youtube.com/watch?v=vPsfZUqI4_0)

\[2\] [Räss, L., Reuber, G., Omlin, S. (2020). Multi-Physics 3-D Inversion on GPU Supercomputers with Julia. JuliaCon Conference, virtual.](https://www.youtube.com/watch?v=1t1AKnnGRqA)

\[3\] [Räss, L., Omlin, S., & Podladchikov, Y. Y. (2019). Porting a Massively Parallel Multi-GPU Application to Julia: a 3-D Nonlinear Multi-Physics Flow Solver. JuliaCon Conference, Baltimore, USA.](https://www.youtube.com/watch?v=b90qqbYJ58Q)

\[4\] [Räss, L., Omlin, S., & Podladchikov, Y. Y. (2019). A Nonlinear Multi-Physics 3-D Solver: From CUDA C + MPI to Julia. PASC19 Conference, Zurich, Switzerland.](https://pasc19.pasc-conference.org/program/schedule/presentation/?id=msa218&sess=sess144)

\[5\] [Besard, T., Foket, C., & De Sutter, B. (2018). Effective Extensible Programming: Unleashing Julia on GPUs. IEEE Transactions on Parallel and Distributed Systems, 30(4), 827-841. doi: 10.1109/TPDS.2018.2872064](https://doi.org/10.1109/TPDS.2018.2872064)

\[6\] [Besard, T., Churavy, V., Edelman, A., & De Sutter B. (2019). Rapid software prototyping for heterogeneous and distributed platforms. Advances in Engineering Software, 132, 29-46. doi: 10.1016/j.advengsoft.2019.02.002](https://doi.org/10.1016/j.advengsoft.2019.02.002)

\[7\] [McCalpin, J. D. (1995). Memory Bandwidth and Machine Balance in Current High Performance Computers. IEEE Computer Society Technical Committee on Computer Architecture (TCCA) Newsletter, December 1995.](https://www.researchgate.net/publication/51992086_Memory_bandwidth_and_machine_balance_in_high_performance_computers)

\[8\] [Räss, L., Duretz, T., & Podladchikov, Y. Y. (2019). Resolving hydromechanical coupling in two and three dimensions: spontaneous channelling of porous fluids owing to de-compaction weakening. Geophysical Journal International, ggz239.](https://doi.org/10.1093/gji/ggz239)
