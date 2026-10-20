# [50-lines multi-xPU example](@id example-multixpu)

This concise multi-xPU 3-D heat diffusion solver uses ParallelStencil in conjunction with [ImplicitGlobalGrid.jl](https://github.com/eth-cscs/ImplicitGlobalGrid.jl) and can be readily deployed on a single CPU thread or on thousands of GPUs/CPUs. Again, a simple boolean `USE_GPU` defines whether it runs on GPU(s) or CPU(s). The solver can be run with any number of processes. [ImplicitGlobalGrid.jl](https://github.com/eth-cscs/ImplicitGlobalGrid.jl) creates automatically an implicit global computational grid based on the number of processes the solver is run with (and based on the process topology, which can be explicitly chosen by the user or automatically determined). Please refer to the [documentation of ImplicitGlobalGrid.jl](https://eth-cscs.github.io/ImplicitGlobalGrid.jl/stable) for more information.

```@eval
Main.mdinclude("diffusion3D_multigpucpu_novis_noperf.jl")
```

The corresponding file can be found [here](https://github.com/omlins/ParallelStencil.jl/blob/main/examples/diffusion3D_multigpucpu_novis_noperf.jl).

Thanks to [ImplicitGlobalGrid.jl](https://github.com/eth-cscs/ImplicitGlobalGrid.jl), only a few function calls had to be added in order to turn the [single GPU/CPU solver](@ref example-single-xpu) into a multi-xPU solver (omitted unmodified lines are represented with `#(...)`):
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

```@raw html
<img src="https://raw.githubusercontent.com/omlins/ParallelStencil.jl/main/examples/movies/diffusion3D_8gpus.gif" alt="3-D heat diffusion on 8 GPUs" style="display: block; margin: auto; width: 100%; max-width: 560px;">
```

The corresponding file, which also hides the communication behind computation with [`@hide_communication`](@ref), can be found [here](https://github.com/omlins/ParallelStencil.jl/blob/main/examples/diffusion3D_multigpucpu_hidecomm.jl).
