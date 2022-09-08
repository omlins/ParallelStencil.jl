const USE_GPU = true
using BenchmarkTools
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2);
else
    @init_parallel_stencil(Threads, Float64, 2);
end

@parallel_indices (ix,iy) function diffusion2D_step!(T2, T, Ci, lam, dt, _dx, _dy)
    tx  = @threadIdx().x + 1
    ty  = @threadIdx().y + 1
    T_l = @sharedMem(eltype(T), (@blockDim().x+2, @blockDim().y+2))
    T_l[tx,ty] = T[ix,iy]
    if (ix>1 && ix<size(T2,1) && iy>1 && iy<size(T2,2))
        if (@threadIdx().x == 1)             T_l[tx-1,ty] = T[ix-1,iy] end
        if (@threadIdx().x == @blockDim().x) T_l[tx+1,ty] = T[ix+1,iy] end
        if (@threadIdx().y == 1)             T_l[tx,ty-1] = T[ix,iy-1] end
        if (@threadIdx().y == @blockDim().y) T_l[tx,ty+1] = T[ix,iy+1] end
        @sync_threads()
        T2[ix,iy] = T_l[tx,ty] + dt*(Ci[ix,iy]*(
                    - ((-lam*(T_l[tx+1,ty] - T_l[tx,ty])*_dx) - (-lam*(T_l[tx,ty] - T_l[tx-1,ty])*_dx))*_dx
                    - ((-lam*(T_l[tx,ty+1] - T_l[tx,ty])*_dy) - (-lam*(T_l[tx,ty] - T_l[tx,ty-1])*_dy))*_dy
                    ));
    end
    return
end

function diffusion2D()
# Physics
lam      = 1.0;                                          # Thermal conductivity
c0       = 2.0;                                          # Heat capacity
lx, ly   = 1.0, 1.0;                                     # Length of computational domain in dimension x and y

# Numerics
nx, ny   = 512*32, 512*32;                               # Number of gridpoints in dimensions x and y
nt       = 100;                                          # Number of time steps
dx       = lx/(nx-1);                                    # Space step in x-dimension
dy       = ly/(ny-1);                                    # Space step in y-dimension
_dx, _dy = 1.0/dx, 1.0/dy;

# Array initializations
T   = @zeros(nx, ny);
T2  = @zeros(nx, ny);
Ci  = @zeros(nx, ny);

# Initial conditions
Ci .= 1/c0;                                              # 1/Heat capacity
T  .= 1.7;
T2 .= T;                                                 # Assign also T2 to get correct boundary conditions.
#
# GPU launch parameters
threads = (32, 8)
blocks  = (nx, ny) .÷ threads
#
# Time loop
dt   = min(dx^2,dy^2)/lam/maximum(Ci)/4.1;               # Time step for 2D Heat diffusion
for it = 1:nt
    if (it == 11) GC.gc(true); global t_tic=time(); end      # Start measuring time.
    @parallel blocks threads shmem=prod(threads.+2)*sizeof(Float64) diffusion2D_step!(T2, T, Ci, lam, dt, _dx, _dy);
    T, T2 = T2, T;
end
time_s = time() - t_tic

# Performance
A_eff = (2*1+1)*1/1e9*nx*ny*sizeof(eltype(T));           # Effective main memory access per iteration [GB] (Lower bound of required memory access: T has to be read and written: 2 whole-array memaccess; Ci has to be read: : 1 whole-array memaccess)
t_it  = time_s/(nt-10);                                  # Execution time per iteration [s]
T_eff = A_eff/t_it;                                      # Effective memory throughput [GB/s]
println("time_s=$time_s t_it=$t_it T_eff=$T_eff");

# Performance
A_eff = (2*1+1)*1/1e9*nx*ny*sizeof(eltype(T));           # Effective main memory access per iteration [GB] (Lower bound of required memory access: T has to be read and written: 2 whole-array memaccess; Ci has to be read: : 1 whole-array memaccess)
t_it = @belapsed begin @parallel $blocks $threads shmem=prod($threads.+2)*sizeof(Float64) diffusion2D_step!($T2, $T, $Ci, $lam, $dt, $_dx, $_dy); end
println("Benchmarktools (min): t_it=$t_it T_eff=$(A_eff/t_it)");

end

diffusion2D()
