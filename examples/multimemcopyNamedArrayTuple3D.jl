const USE_GPU = true
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3);
else
    @init_parallel_stencil(Threads, Float64, 3);
end

@parallel function copy3D!(V2::Data.NamedArrayTuple, V::Data.NamedArrayTuple, dV::Data.NamedArrayTuple)
    @all(V2.x) = @all(V.x) + @all(dV.x)
    @all(V2.y) = @all(V.y) + @all(dV.y)
    @all(V2.z) = @all(V.z) + @all(dV.z)
    return
end

function memcopy3D()
# Numerics
nx, ny, nz = 512, 512, 512;                              # Number of gridpoints in dimensions x, y and z
nt  = 100;                                               # Number of time steps

# Array initializations
V  = (x = @zeros(nx+1, ny, nz), y  = @zeros(nx, ny+1, nz), z  = @zeros(nx, ny, nz+1))
V2 = (x = @zeros(nx+1, ny, nz), y  = @zeros(nx, ny+1, nz), z  = @zeros(nx, ny, nz+1))
dV = (x = @zeros(nx+1, ny, nz), y  = @zeros(nx, ny+1, nz), z  = @zeros(nx, ny, nz+1))

# Initial conditions
dV.x .= 1/2.0; dV.y .= 1/2.0; dV.z .= 1/2.0;
V.x  .= 1.7;   V.y  .= 1.7;   V.z  .= 1.7;
V2.x .= V.x;   V2.y .= V.y;   V2.z .= V.z;

# Time loop
for it = 1:nt
    if (it == 11) global t0=time(); end  # Start measuring time.
    @parallel copy3D!(V2, V, dV);
    V, V2 = V2, V;
end
time_s=time()-t0

# Performance
A_eff = (2*3+3)*1/1e9*nx*ny*nz*sizeof(Data.Number);      # Effective main memory access per iteration [GB] (Lower bound of required memory access: T has to be read and written: 2 whole-array memaccess; Ci has to be read: : 1 whole-array memaccess)
t_it  = time_s/(nt-10);                                  # Execution time per iteration [s]
T_eff = A_eff/t_it;                                      # Effective memory throughput [GB/s]
println("time_s=$time_s T_eff=$T_eff");
end

memcopy3D()
