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

@parallel function diffusion3D_step!(T2, T, Ci, lam, dt, _dx, _dy, _dz)
    @inn(T2) = @inn(T) + dt*(lam*@inn(Ci)*(@d2_xi(T)*_dx^2 + @d2_yi(T)*_dy^2 + @d2_zi(T)*_dz^2));
    return
end

@parallel function diffusion3D_step!(T2, T, Ci, lam, dt, _dx, _dy, _dz)
    @inn(T2) = @inn(T) + dt*(lam*@inn(Ci)*(@d2_xi(T)*_dx^2 + @d2_yi(T)*_dy^2 + @d2_zi(T)*_dz^2));
    return
end

function diffusion3D()
# Physics
lam        = 1.0;                                        # Thermal conductivity
c0         = 2.0;                                        # Heat capacity
lx, ly, lz = 1.0, 1.0, 1.0;                              # Length of computational domain in dimension x, y and z

# Numerics
nx, ny, nz = 512, 512, 512;                              # Number of gridpoints in dimensions x, y and z
nt         = 100;                                        # Number of time steps
me, dims   = init_global_grid(nx, ny, nz);
dx         = lx/(nx_g()-1);                              # Space step in x-dimension
dy         = ly/(ny_g()-1);                              # Space step in y-dimension
dz         = lz/(nz_g()-1);                              # Space step in z-dimension
_dx, _dy, _dz = 1.0/dx, 1.0/dy, 1.0/dz;

# Array initializations
T   = @zeros(nx, ny, nz);
T2  = @zeros(nx, ny, nz);
Ci  = @zeros(nx, ny, nz);

# Initial conditions
Ci .= 1/c0;                                              # 1/Heat capacity
T  .= 1.7;
T2 .= T;                                                 # Assign also T2 to get correct boundary conditions.

# Time loop
dt   = min(dx^2,dy^2,dz^2)/lam/maximum(Ci)/8.1;          # Time step for 3D Heat diffusion
for it = 1:nt
    if (it == 11) tic(); end                             # Start measuring time.
    @hide_communication (16, 2, 2) begin
        @parallel diffusion3D_step!(T2, T, Ci, lam, dt, _dx, _dy, _dz);
        T2[  1,:,:] .= 1.0;
        T2[end,:,:] .= 1.0;
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

finalize_global_grid();
end

diffusion3D()
