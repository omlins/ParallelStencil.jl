# ParallelStencil.jl

ParallelStencil empowers domain scientists to write architecture-agnostic high-level code for parallel high-performance stencil computations on GPUs and CPUs. Performance similar to CUDA C can be achieved \[[1][JuliaCon20a]\]:

![Performance ParallelStencil vs CUDA C](docs/images/fig_Teff_HM3D_Julia_vs_CUDA_C.png)

ParallelStencil relies on the native kernel programming capabilities of [CUDA.jl] and on [Base.Threads] for high-performance computations on GPUs and CPUs, respectively. It is seamlessly interoperable with [ImplicitGlobalGrid.jl], which renders the distributed parallelization of stencil-based GPU and CPU applications on a regular staggered grid almost trivial and enables close to ideal weak scaling of real-world applications on thousands of GPUs \[[1][JuliaCon20a], [2][JuliaCon20b], [3][JuliaCon19], [4][PASC19]\]. Moreover, ParallelStencil enables hiding communication behind computation with as simple macro call and without any particular restrictions on the package used for communication. ParallelStencil has been designed in conjunction with [ImplicitGlobalGrid.jl] for simplest possible usage by domain-scientists, rendering fast and interactive development of massively scalable high performance Multi-GPU applications readily accessible to them. Furthermore, we have developed a self-contained approach for "Solving Nonlinear Multi-Physics on GPU Supercomputers with Julia" relying on ParallelStencil and [ImplicitGlobalGrid.jl] \[[1][JuliaCon20a]\].

A particularity of ParallelStencil is that it enables writing a single high-level Julia code that can be deployed both on a CPU or a GPU. In conjuction with [ImplicitGlobalGrid.jl] the same Julia code can even run on a single CPU thread or on thousands of GPUs/CPUs.

## Contents
* [Parallelization with one macro call](#parallelization-with-one-macro-call)
* [Stencil computations with math-close notation](#stencil-computations-with-math-close-notation)
* [50-lines example deployable on GPU and CPU](#50-lines-example-deployable-on-GPU-and-CPU)
* [50-lines Multi-GPU/CPU example](#50-lines-multi-gpucpu-example)
* [Seamless interoperability with communication packages and hiding communication](#seamless-interoperability-with-communication-packages-and-hiding-communication)
* [Module documentation callable from the Julia REPL / IJulia](#module-documentation-callable-from-the-julia-repl--ijulia)
* [Dependencies](#dependencies)
* [Installation](#installation)
* [References](#references)

## Parallelization with one macro call
A simple call to `@parallel` is enough to parallelize a function and to launch it. The package used underneath for parallelization is defined in a call to `@init_parallel_stencil` beforehand. Supported are [CUDA.jl] for running on GPU and [Base.Threads] for CPU. The following example outlines how to run parallel computations on a GPU using the native kernel programming capabilities of [CUDA.jl] underneath (omitted lines are represented with `#(...)`, omitted arguments with `...`):
```julia
#(...)
@init_parallel_stencil(CUDA,...);
#(...)
@parallel function diffusion3D_step!(...)
    #(...)
end
#(...)
@parallel diffusion3D_step!(...)
```
Note that arrays are automatically allocated on the hardware chosen for the computations (GPU or CPU) when using the provided allocation macros:
- `@zeros`
- `@ones`
- `@rand`

## Stencil computations with math-close notation
ParallelStencil provides submodules for computing finite differences in 1-D, 2-D and 3-D with a math-close notation (`FiniteDifferences1D`, `FiniteDifferences2D` and `FiniteDifferences3D`). Custom macros to extend the finite differences submodules or for other stencil-based numerical methods can be readily plugged in. The following example shows a complete function for computing a time step of a 3-D heat diffusion solver using `FiniteDifferences3D`.
```julia
#(...)
using ParallelStencil.FiniteDifferences3D
#(...)
@parallel function diffusion3D_step!(T2, T, Ci, lam, dt, dx, dy, dz)
    @inn(T2) = @inn(T) + dt*(lam*@inn(Ci)*(@d2_xi(T)/dx^2 + @d2_yi(T)/dy^2 + @d2_zi(T)/dz^2));
    return
end
```
The macros used in this example are described in the [Module documentation callable from the Julia REPL / IJulia](#module-documentation-callable-from-the-julia-repl--ijulia):
```julia-repl
julia> using ParallelStencil.FiniteDifferences3D
julia>?
help?> @inn
  @inn(A): Select the inner elements of A. Corresponds to A[2:end-1,2:end-1,2:end-1].

help?> @d2_xi
  @d2_xi(A): Compute the 2nd order differences between adjacent elements of A along the along dimension x and select the inner elements of A in the remaining dimensions. Corresponds to @inn_yz(@d2_xa(A)).
```
Note that`@d2_yi` and `@d2_zi` perform the analogue operations as `@d2_xi` along the dimension y and z, respectively.

Type `?FiniteDifferences3D` in the [Julia REPL] to explore all provided macros.

## 50-lines example deployable on GPU and CPU
This concise 3-D heat diffusion solver uses ParallelStencil and a a simple boolean `USE_GPU` defines whether it runs on GPU or CPU (the environment variable [JULIA_NUM_THREADS] defines how many cores are used in the latter case):

```julia
const USE_GPU = true
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3);
else
    @init_parallel_stencil(Threads, Float64, 3);
end

@parallel function diffusion3D_step!(T2, T, Ci, lam, dt, dx, dy, dz)
    @inn(T2) = @inn(T) + dt*(lam*@inn(Ci)*(@d2_xi(T)/dx^2 + @d2_yi(T)/dy^2 + @d2_zi(T)/dz^2));
    return
end

function diffusion3D()
# Physics
lam        = 1.0;                                        # Thermal conductivity
cp_min     = 1.0;                                        # Minimal heat capacity
lx, ly, lz = 10.0, 10.0, 10.0;                           # Length of domain in dimensions x, y and z.

# Numerics
nx, ny, nz = 256, 256, 256;                              # Number of gridpoints dimensions x, y and z.
nt         = 100;                                        # Number of time steps
dx         = lx/(nx-1);                                  # Space step in x-dimension
dy         = ly/(ny-1);                                  # Space step in y-dimension
dz         = lz/(nz-1);                                  # Space step in z-dimension

# Array initializations
T   = @zeros(nx, ny, nz);
T2  = @zeros(nx, ny, nz);
Ci  = @zeros(nx, ny, nz);

# Initial conditions (heat capacity and temperature with two Gaussian anomalies each)
Ci .= 1.0./( cp_min .+ Data.Array([5*exp(-(((ix-1)*dx-lx/1.5))^2-(((iy-1)*dy-ly/2))^2-(((iz-1)*dz-lz/1.5))^2) +
                                   5*exp(-(((ix-1)*dx-lx/3.0))^2-(((iy-1)*dy-ly/2))^2-(((iz-1)*dz-lz/1.5))^2) for ix=1:size(T,1), iy=1:size(T,2), iz=1:size(T,3)]) )
T  .= Data.Array([100*exp(-(((ix-1)*dx-lx/2)/2)^2-(((iy-1)*dy-ly/2)/2)^2-(((iz-1)*dz-lz/3.0)/2)^2) +
                   50*exp(-(((ix-1)*dx-lx/2)/2)^2-(((iy-1)*dy-ly/2)/2)^2-(((iz-1)*dz-lz/1.5)/2)^2) for ix=1:size(T,1), iy=1:size(T,2), iz=1:size(T,3)])
T2 .= T;                                                 # Assign also T2 to get correct boundary conditions.

# Time loop
dt = min(dx^2,dy^2,dz^2)*cp_min/lam/8.1;                 # Time step for the 3D Heat diffusion
for it = 1:nt
    @parallel diffusion3D_step!(T2, T, Ci, lam, dt, dx, dy, dz);
    T, T2 = T2, T;
end

end

diffusion3D()
```

The corresponding file can be found [here](/examples/diffusion3D_novis_noperf.jl).

## 50-lines Multi-GPU/CPU example
This concise Multi-GPU/CPU 3-D heat diffusion solver uses ParallelStencil in conjunction with [ImplicitGlobalGrid.jl] and can be readily deployed on on a single CPU thread or on thousands of GPUs/CPUs. Again, a simple boolean `USE_GPU` defines whether it runs on GPU(s) or CPU(s) ([JULIA_NUM_THREADS] defines how many cores are used in the latter case). The solver can be run with any number of processes. [ImplicitGlobalGrid.jl] creates automatically an implicit global computational grid based on the number of processes the solver is run with (and based on the process topology, which can be explicitly chosen by the user or automatically determined). Please refer to the documentation of [ImplicitGlobalGrid.jl] for more information.

```julia
const USE_GPU = true
using ImplicitGlobalGrid
import MPI
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3);
else
    @init_parallel_stencil(Threads, Float64, 3);
end

@parallel function diffusion3D_step!(T2, T, Ci, lam, dt, dx, dy, dz)
    @inn(T2) = @inn(T) + dt*(lam*@inn(Ci)*(@d2_xi(T)/dx^2 + @d2_yi(T)/dy^2 + @d2_zi(T)/dz^2));
    return
end

function diffusion3D()
# Physics
lam        = 1.0;                                        # Thermal conductivity
cp_min     = 1.0;                                        # Minimal heat capacity
lx, ly, lz = 10.0, 10.0, 10.0;                           # Length of domain in dimensions x, y and z.

# Numerics
nx, ny, nz = 256, 256, 256;                              # Number of gridpoints dimensions x, y and z.
nt         = 100;                                        # Number of time steps
init_global_grid(nx, ny, nz);
dx         = lx/(nx_g()-1);                              # Space step in x-dimension
dy         = ly/(ny_g()-1);                              # Space step in y-dimension
dz         = lz/(nz_g()-1);                              # Space step in z-dimension

# Array initializations
T   = @zeros(nx, ny, nz);
T2  = @zeros(nx, ny, nz);
Ci  = @zeros(nx, ny, nz);

# Initial conditions (heat capacity and temperature with two Gaussian anomalies each)
Ci .= 1.0./( cp_min .+ Data.Array([5*exp(-((x_g(ix,dx,Ci)-lx/1.5))^2-((y_g(iy,dy,Ci)-ly/2))^2-((z_g(iz,dz,Ci)-lz/1.5))^2) +
                                   5*exp(-((x_g(ix,dx,Ci)-lx/3.0))^2-((y_g(iy,dy,Ci)-ly/2))^2-((z_g(iz,dz,Ci)-lz/1.5))^2) for ix=1:size(T,1), iy=1:size(T,2), iz=1:size(T,3)]) )
T  .= Data.Array([100*exp(-((x_g(ix,dx,T)-lx/2)/2)^2-((y_g(iy,dy,T)-ly/2)/2)^2-((z_g(iz,dz,T)-lz/3.0)/2)^2) +
                   50*exp(-((x_g(ix,dx,T)-lx/2)/2)^2-((y_g(iy,dy,T)-ly/2)/2)^2-((z_g(iz,dz,T)-lz/1.5)/2)^2) for ix=1:size(T,1), iy=1:size(T,2), iz=1:size(T,3)])
T2 .= T;                                                 # Assign also T2 to get correct boundary conditions.

# Time loop
dt = min(dx^2,dy^2,dz^2)*cp_min/lam/8.1;                 # Time step for the 3D Heat diffusion
for it = 1:nt
    @parallel diffusion3D_step!(T2, T, Ci, lam, dt, dx, dy, dz);
    update_halo!(T2);
    T, T2 = T2, T;
end

finalize_global_grid();
end

diffusion3D()
```
The corresponding file can be found [here](/examples/diffusion3D_multigpucpu_novis_noperf.jl).

Thanks to [ImplicitGlobalGrid.jl], only a few function calls had to be added in order to turn the previous single GPU/CPU solver into a Multi-GPU/CPU solver (omitted unmodified lines are represented with `#(...)`):
```julia
#(...)
using ImplicitGlobalGrid
#(...)
function diffusion3D()
# Physics
#(...)
# Numerics
#(...)
init_global_grid(nx, ny, nz);
dx  = lx/(nx_g()-1);                                     # Space step in x-dimension
dy  = ly/(ny_g()-1);                                     # Space step in y-dimension
dz  = lz/(nz_g()-1);                                     # Space step in z-dimension

# Array initializations
#(...)

# Initial conditions (heat capacity and temperature with two Gaussian anomalies each)
Ci .= 1.0./( cp_min .+ Data.Array([5*exp(-((x_g(ix,dx,Ci)-lx/1.5))^2-((y_g(iy,dy,Ci)-ly/2))^2-((z_g(iz,dz,Ci)-lz/1.5))^2) +
                                   5*exp(-((x_g(ix,dx,Ci)-lx/3.0))^2-((y_g(iy,dy,Ci)-ly/2))^2-((z_g(iz,dz,Ci)-lz/1.5))^2) for ix=1:size(T,1), iy=1:size(T,2), iz=1:size(T,3)]) )
T  .= Data.Array([100*exp(-((x_g(ix,dx,T)-lx/2)/2)^2-((y_g(iy,dy,T)-ly/2)/2)^2-((z_g(iz,dz,T)-lz/3.0)/2)^2) +
                   50*exp(-((x_g(ix,dx,T)-lx/2)/2)^2-((y_g(iy,dy,T)-ly/2)/2)^2-((z_g(iz,dz,T)-lz/1.5)/2)^2) for ix=1:size(T,1), iy=1:size(T,2), iz=1:size(T,3)])

# Time loop
#(...)
for it = 1:nt
    #(...)
    update_halo!(T2);
    #(...)
end

finalize_global_grid();
end

diffusion3D()
```

Here is the resulting movie when running the application on 8 GPUs, solving 3-D heat diffusion with heterogeneous heat capacity (two Gaussian anomalies) on a global computational grid of size 510x510x510 grid points. It shows the x-z-dimension plane in the middle of the dimension y:

![3-D heat diffusion](examples/movies/diffusion3D_8gpus.gif)

## Seamless interoperability with communication packages and hiding communication
The previous Multi-GPU/CPU example shows that ParallelStencil is seamlessly interoperability with [ImplicitGlobalGrid.jl]. The same is a priori true for any communication package that allows to explicitly decide when the required communication occurs; an example is [MPI.jl] (besides, [MPI.jl] is also seamlessly interoperable with [ImplicitGlobalGrid.jl] and can extend its functionality).

Moreover, ParallelStencil enables hiding communication behind computation with as simple call to `@hide_communication`. In the following example, the communication performed by `update_halo!` (from the package [ImplicitGlobalGrid.jl]) is hidden behind the computations done with by `@parallel diffusion3D_step!`:
```julia
@hide_communication (16, 2, 2) begin
    @parallel diffusion3D_step!(T2, Te Ci, lam, dt, dx, dy, dz);
    update_halo!(T2);
end
```
This enables close to ideal weak scaling of real-world applications on thousands of GPUs/CPUs \[[1][JuliaCon20a], [2][JuliaCon20b]\]. Type `?@hide_communication` in the [Julia REPL] to obtain an explanation of the arguments.

## Module documentation callable from the Julia REPL / IJulia
The module documentation can be called from the [Julia REPL] or in [IJulia]:
```julia-repl
julia> using ParallelStencil
julia>?
help?> ParallelStencil
search: ParallelStencil @init_parallel_stencil

  Module ParallelStencil

  Enables domain scientists to write high-level code for parallel high-performance stencil computations that can be deployed on both GPUs and CPUs.

  Macros and functions
  ≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡

    •    @init_parallel_stencil

    •    @parallel

    •    @hide_communication

    •    @zeros

    •    @ones

    •    @rand

  │ Advanced
  │
  │    •    @parallel_indices
  │
  │    •    @parallel_async
  │
  │    •    @synchronize

  Submodules
  ≡≡≡≡≡≡≡≡≡≡≡≡

    •    ParallelStencil.FiniteDifferences1D

    •    ParallelStencil.FiniteDifferences2D

    •    ParallelStencil.FiniteDifferences3D

    •    Data

  To see a description of a function or a macro type ?<functionname> or ?<macroname> (including the @), respectively.
```

## Dependencies
ParallelStencil relies on the Julia CUDA package ([CUDA.jl] \[[5][Julia CUDA paper 1], [6][Julia CUDA paper 2]\]) and [MacroTools.jl].

## Installation
ParallelStencil may be installed directly with the [Julia package manager](https://docs.julialang.org/en/v1/stdlib/Pkg/index.html) from the REPL:
```julia-repl
julia>]
  pkg> add https://github.com/omlins/ParallelStencil.jl
```

## References
\[1\] [Omlin, S., Räss, L., Kwasniewski, G., Malvoisin, B., & Podladchikov, Y. Y. (2020). Solving Nonlinear Multi-Physics on GPU Supercomputers with Julia. JuliaCon Conference, virtual.][JuliaCon20a]

\[2\] [Räss, L., Reuber, G., Omlin, S. (2020). Multi-Physics 3-D Inversion on GPU Supercomputers with Julia. JuliaCon Conference, virtual.][JuliaCon20b]

\[3\] [Räss, L., Omlin, S., & Podladchikov, Y. Y. (2019). Porting a Massively Parallel Multi-GPU Application to Julia: a 3-D Nonlinear Multi-Physics Flow Solver. JuliaCon Conference, Baltimore, USA.][JuliaCon19]

\[4\] [Räss, L., Omlin, S., & Podladchikov, Y. Y. (2019). A Nonlinear Multi-Physics 3-D Solver: From CUDA C + MPI to Julia. PASC19 Conference, Zurich, Switzerland.][PASC19]

\[5\] [Besard, T., Foket, C., & De Sutter, B. (2018). Effective Extensible Programming: Unleashing Julia on GPUs. IEEE Transactions on Parallel and Distributed Systems, 30(4), 827-841. doi: 10.1109/TPDS.2018.2872064][Julia CUDA paper 1]

\[6\] [Besard, T., Churavy, V., Edelman, A., & De Sutter B. (2019). Rapid software prototyping for heterogeneous and distributed platforms. Advances in Engineering Software, 132, 29-46. doi: 10.1016/j.advengsoft.2019.02.002][Julia CUDA paper 2]

[JuliaCon20a]: https://www.youtube.com/watch?v=vPsfZUqI4_0
[JuliaCon20b]: https://www.youtube.com/watch?v=1t1AKnnGRqA
[JuliaCon19]: https://pretalx.com/juliacon2019/talk/LGHLC3/
[PASC19]: https://pasc19.pasc-conference.org/program/schedule/presentation/?id=msa218&sess=sess144
[Base.Threads]: https://docs.julialang.org/en/v1/base/multi-threading/
[ImplicitGlobalGrid.jl]: https://github.com/eth-cscs/ImplicitGlobalGrid.jl
[JULIA_NUM_THREADS]:https://docs.julialang.org/en/v1.0.0/manual/environment-variables/#JULIA_NUM_THREADS-1
[MPI.jl]: https://github.com/JuliaParallel/MPI.jl
[CUDA.jl]: https://github.com/JuliaGPU/CUDA.jl
[MacroTools.jl]: https://github.com/FluxML/MacroTools.jl
[Julia CUDA paper 1]: https://doi.org/10.1109/TPDS.2018.2872064
[Julia CUDA paper 2]: https://doi.org/10.1016/j.advengsoft.2019.02.002
[Julia REPL]: https://docs.julialang.org/en/v1/stdlib/REPL/
[IJulia]: https://github.com/JuliaLang/IJulia.jl
