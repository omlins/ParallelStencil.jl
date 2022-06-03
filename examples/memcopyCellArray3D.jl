const USE_GPU = true
using CellArrays, StaticArrays
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3);
else
    @init_parallel_stencil(Threads, Float64, 3);
end

# @parallel function copy3D!(T2, T, Ci)
#     @all(T2) = @all(T) + @all(Ci);
#     return
# end

@parallel_indices (ix,iy,iz) function copy3D!(T2::Data.CellArray, T::Data.CellArray, Ci::Data.CellArray)
    T2[ix,iy,iz] = T[ix,iy,iz] + Ci[ix,iy,iz];
    return
end

struct Stiffness{T} <: FieldArray{Tuple{4,4}, T, 4}
    xxxx::T
    yxxx::T
    xyxx::T
    yyxx::T
    xxyx::T
    yxyx::T
    xyyx::T
    yyyx::T
    xxxy::T
    yxxy::T
    xyxy::T
    yyxy::T
    xxyy::T
    yxyy::T
    xyyy::T
    yyyy::T
end


function memcopy3D()
# Numerics
nx, ny, nz = 128, 128, 1024                                                     # Number of gridpoints in dimensions x, y and z
celldims   = (4,4)
nt         = 100;                                                               # Number of time steps

# Array initializations
T  = @zeros(nx, ny, nz, celldims=celldims);
T2 = @zeros(nx, ny, nz, celldims=celldims);
Ci = @zeros(nx, ny, nz, celldims=celldims);
# T  = @zeros(nx, ny, nz, celltype=Stiffness{Data.Number});
# T2 = @zeros(nx, ny, nz, celltype=Stiffness{Data.Number});
# Ci = @zeros(nx, ny, nz, celltype=Stiffness{Data.Number});

# Initial conditions
@fill!(Ci, 0.5);
@fill!(T, 1.7);
copy!(T2.data, T.data);

# Time loop
for it = 1:nt
    if (it == 11) global t0=time(); end  # Start measuring time.
    @parallel copy3D!(T2, T, Ci);
    T, T2 = T2, T;
end
time_s=time()-t0

# Performance
A_eff = (2*1+1)*1/1e9*nx*ny*nz*prod(celldims)*sizeof(Data.Number);      # Effective main memory access per iteration [GB] (Lower bound of required memory access: T has to be read and written: 2 whole-array memaccess; Ci has to be read: : 1 whole-array memaccess)
t_it  = time_s/(nt-10);                                                 # Execution time per iteration [s]
T_eff = A_eff/t_it;                                                     # Effective memory throughput [GB/s]
println("time_s=$time_s T_eff=$T_eff");
end

memcopy3D()
