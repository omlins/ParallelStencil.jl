const USE_GPU = true
using ImplicitGlobalGrid
import MPI
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
@static if USE_GPU
    @init_parallel_stencil(package=CUDA, ndims=3);
else
    @init_parallel_stencil(package=Threads, ndims=3);
end

@parallel function diffusion3D_step!(T2::Data.Array{_T}, T::Data.Array{_T}, Ci::Data.Array{_T}, lam::_T, dt::_T, _dx::_T, _dy::_T, _dz::_T) where _T <: ParallelStencil.PSNumber
    @inn(T2) = @inn(T) + dt*(lam*@inn(Ci)*(@d2_xi(T)*_dx^2 + @d2_yi(T)*_dy^2 + @d2_zi(T)*_dz^2));
    return
end

function diffusion3D(lam::_T, c0::_T, lx::_T, ly::_T, lz::_T) where _T <: ParallelStencil.PSNumber

# Numerics
nx, ny, nz = 512, 512, 512;                              # Number of gridpoints in dimensions x, y and z
nt         = 100;                                        # Number of time steps
me, dims   = init_global_grid(nx, ny, nz);
dx         = lx/(nx_g()-1);                              # Space step in x-dimension
dy         = ly/(ny_g()-1);                              # Space step in y-dimension
dz         = lz/(nz_g()-1);                              # Space step in z-dimension
_dx, _dy, _dz = 1.0/dx, 1.0/dy, 1.0/dz;

# Array initializations
T   = @zeros(_T, nx, ny, nz);
T2  = @zeros(_T, nx, ny, nz);
Ci  = @zeros(_T, nx, ny, nz);

# Initial conditions
Ci .= 1/c0;                                              # 1/Heat capacity
T  .= 1.7;
T2 .= T;                                                 # Assign also T2 to get correct boundary conditions.

# Time loop
dt   = min(dx^2,dy^2,dz^2)/lam/maximum(Ci)/8.1;          # Time step for 3D Heat diffusion
lam, dt, _dx, _dy, _dz = _T.((lam, dt, _dx, _dy, _dz))   # Convert all scalars to the required numbertype.
for it = 1:nt
    if (it == 11) tic(); end                             # Start measuring time.
    @parallel diffusion3D_step!(T2, T, Ci, lam, dt, _dx, _dy, _dz);
    update_halo!(T2);
    T, T2 = T2, T;
end
time_s = toc()

# Performance
A_eff = (2*1+1)*1/1e9*nx*ny*nz*sizeof(_T);               # Effective main memory access per iteration [GB] (Lower bound of required memory access: T has to be read and written: 2 whole-array memaccess; Ci has to be read: : 1 whole-array memaccess)
t_it  = time_s/(nt-10);                                  # Execution time per iteration [s]
T_eff = A_eff/t_it;                                      # Effective memory throughput [GB/s]
if (me==0) println("time_s=$time_s T_eff=$T_eff"); end

finalize_global_grid();
end


################################################################################
# CALL OF THE FUNCTION

# Physics
lam        = 1.0;                                        # Thermal conductivity
c0         = 2.0;                                        # Heat capacity
lx, ly, lz = 1.0, 1.0, 1.0;                              # Length of computational domain in dimension x, y and z

lam, c0, lx, ly, lz = Float64.((lam, c0, lx, ly, lz))    # Convert all scalars to the desired numbertype.

diffusion3D(lam, c0, lx, ly, lz)
