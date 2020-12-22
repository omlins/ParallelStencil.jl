const USE_GPU = true
using ImplicitGlobalGrid, Plots
import MPI
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3);
else
    @init_parallel_stencil(Threads, Float64, 3);
end

@parallel function diffusion3D_step!(T2, T, Ci, lam, dt, _dx, _dy, _dz)
    @inn(T2) = @inn(T) + dt*(lam*@inn(Ci)*(@d2_xi(T)*_dx^2 + @d2_yi(T)*_dy^2 + @d2_zi(T)*_dz^2));
    return
end

function diffusion3D()
# Physics
lam        = 1.0;                                       # Thermal conductivity
cp_min     = 1.0;                                       # Minimal heat capacity
lx, ly, lz = 10.0, 10.0, 10.0;                          # Length of computational domain in dimension x, y and z

# Numerics
nx, ny, nz = 256, 256, 256;                             # Number of gridpoints in dimensions x, y and z
nt         = 100000;                                    # Number of time steps
me, dims   = init_global_grid(nx, ny, nz);
dx         = lx/(nx_g()-1);                             # Space step in dimension x
dy         = ly/(ny_g()-1);                             # ...        in dimension y
dz         = lz/(nz_g()-1);                             # ...        in dimension z
_dx, _dy, _dz = 1.0/dx, 1.0/dy, 1.0/dz;

# Array initializations
T   = @zeros(nx, ny, nz);
T2  = @zeros(nx, ny, nz);
Ci  = @zeros(nx, ny, nz);

# Initial conditions (heat capacity and temperature with two Gaussian anomalies each)
Ci .= 1.0./( cp_min .+ Data.Array([5*exp(-((x_g(ix,dx,Ci)-lx/1.5))^2-((y_g(iy,dy,Ci)-ly/2))^2-((z_g(iz,dz,Ci)-lz/1.5))^2) +
                                   5*exp(-((x_g(ix,dx,Ci)-lx/3.0))^2-((y_g(iy,dy,Ci)-ly/2))^2-((z_g(iz,dz,Ci)-lz/1.5))^2) for ix=1:size(T,1), iy=1:size(T,2), iz=1:size(T,3)]) )
T  .= Data.Array([100*exp(-((x_g(ix,dx,T)-lx/2)/2)^2-((y_g(iy,dy,T)-ly/2)/2)^2-((z_g(iz,dz,T)-lz/3.0)/2)^2) +
                   50*exp(-((x_g(ix,dx,T)-lx/2)/2)^2-((y_g(iy,dy,T)-ly/2)/2)^2-((z_g(iz,dz,T)-lz/1.5)/2)^2) for ix=1:size(T,1), iy=1:size(T,2), iz=1:size(T,3)])
T2 .= T;                                                # Assign also T2 to get correct boundary conditions.

# Preparation of visualisation
gr()
ENV["GKSwstype"]="nul"
anim = Animation();
nx_v = (nx-2)*dims[1];
ny_v = (ny-2)*dims[2];
nz_v = (nz-2)*dims[3];
T_v  = zeros(nx_v, ny_v, nz_v);
T_nohalo = zeros(nx-2, ny-2, nz-2);

# Time loop
dt = min(dx^2,dy^2,dz^2)*cp_min/lam/8.1;                                                  # Time step for the 3D Heat diffusion
for it = 1:nt
    if (it == 11) tic(); end                                                              # Start measuring time.
    if mod(it, 1000) == 1                                                                 # Visualize only every 1000th time step
        T_nohalo .= T[2:end-1,2:end-1,2:end-1];                                           # Copy data to CPU removing the halo.
        gather!(T_nohalo, T_v)                                                            # Gather data on process 0 (could be interpolated/sampled first)
        if (me==0) heatmap(transpose(T_v[:,ny_v÷2,:]), aspect_ratio=1); frame(anim); end  # Visualize it on process 0.
    end
    @hide_communication (16, 2, 2) begin
        @parallel diffusion3D_step!(T2, T, Ci, lam, dt, _dx, _dy, _dz);
        update_halo!(T2);
    end
    T, T2 = T2, T;
end
time_s = toc()

# Performance
A_eff = (2*1+1)*1/1e9*nx*ny*nz*sizeof(Data.Number);      # Effective main memory access per iteration [GB] (Lower bound of required memory access: T has to be read and written: 2 whole-array memaccess; Ci has to be read: : 1 whole-array memaccess)
t_it  = time_s/(nt-10);                                  # Execution time per iteration [s]
T_eff = A_eff/t_it;                                      # Effective memory throughput [GB/s]
if (me==0) println("time_s=$time_s T_eff=$T_eff"); end

# Postprocessing
if (me==0) gif(anim, "diffusion3D.gif", fps = 15) end    # Create a gif movie on process 0.
if (me==0) mp4(anim, "diffusion3D.mp4", fps = 15) end    # Create a mp4 movie on process 0.
finalize_global_grid();                                  # Finalize the implicit global grid
end

diffusion3D()
