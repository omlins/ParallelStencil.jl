const USE_GPU = true
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3);
else
    @init_parallel_stencil(Threads, Float64, 3);
end

@parallel function copy3D!(T2::Data.CellArray, T::Data.CellArray, Ci::Data.CellArray)
    @all(T2) = @all(T) + @all(Ci);
    return
end

@parallel_indices (ix,iy,iz) function copy3D_explicit!(T2::Data.CellArray, T::Data.CellArray, Ci::Data.CellArray)
    T2[ix,iy,iz] = T[ix,iy,iz] + Ci[ix,iy,iz]
    return
end

using CellArrays
@parallel_indices (ix,iy,iz) function copy3D_large_cells!(T2::Data.CellArray, T::Data.CellArray, Ci::Data.CellArray)
    for cy = 1:cellsize(T2,2), cx = 1:cellsize(T2,1)
        field(T2,cx,cy)[ix,iy,iz] = field(T,cx,cy)[ix,iy,iz] + field(Ci,cx,cy)[ix,iy,iz]
    end
    return
end


@CellType Cell4x4 fieldnames=(xxxx, yxxx, xyxx, yyxx, xxyx, yxyx, xyyx, yyyx, xxxy, yxxy, xyxy, yyxy, xxyy, yxyy, xyyy, yyyy) dims=(4,4)
@CellType TCell4x4 fieldnames=(xxxx, yxxx, xyxx, yyxx, xxyx, yxyx, xyyx, yyyx, xxxy, yxxy, xyxy, yyxy, xxyy, yxyy, xyyy, yyyy) dims=(4,4) parametric=true


function memcopy3D()
# Numerics
nx, ny, nz = 128,128,1024 #CPU:128,128,128                                      # Number of gridpoints in dimensions x, y and z
celldims   = (4,4)
nt         = 100;                                                               # Number of time steps

# Array initializations
T  = @zeros(nx, ny, nz, celltype=Cell4x4);  # or: celltype=TCell4x4{Float64}
T2 = @zeros(nx, ny, nz, celltype=Cell4x4);  # or: celltype=TCell4x4{Float64}
Ci = @zeros(nx, ny, nz, celldims=celldims);

# Initial conditions
@fill!(Ci, 0.5);
@fill!(T, 1.7);
copy!(T2.data, T.data);

# Time loop
for it = 1:nt
    if (it == 11) global t0=time(); end  # Start measuring time.
    @parallel copy3D!(T2, T, Ci);        # or: @parallel copy3D_explicit!(T2, T, Ci);  or: @parallel copy3D_large_cells!(T2, T, Ci);
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
