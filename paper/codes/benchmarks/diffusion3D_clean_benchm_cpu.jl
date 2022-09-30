using BenchmarkTools
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
@init_parallel_stencil(Threads, Float64, 3)

@parallel function step!(
    T2, T, Ci, lam, dt, _dx, _dy, _dz
    )
    @inn(T2) = @inn(T) + dt*(
        lam*@inn(Ci)*(@d2_xi(T)*_dx^2 + 
                      @d2_yi(T)*_dy^2 + 
                      @d2_zi(T)*_dz^2 ) )
    return
end

function diffusion3D()
    # Physics
    lam      = 1.0        # Thermal conductivity
    c0       = 2.0        # Heat capacity
    lx=ly=lz = 1.0        # Domain length x|y|z

    # Numerics
    nx=ny=nz = 512        # Nb gridpoints x|y|z
    nt       = 100        # Nb time steps
    dx       = lx/(nx-1)  # Space step in x
    dy       = ly/(ny-1)  # Space step in y
    dz       = lz/(nz-1)  # Space step in z
    _dx, _dy, _dz = 1.0/dx, 1.0/dy, 1.0/dz

    # Array initializations
    T  = @zeros(nx,ny,nz) # Temperature
    T2 = @zeros(nx,ny,nz) # Temperature (2nd)
    Ci = @zeros(nx,ny,nz) # 1/Heat capacity

    # Initial conditions
    Ci .= 1/c0
    T  .= 1.7
    T2 .= T

    # Time loop
    dt = min(dx^2,dy^2,dz^2)/lam/maximum(Ci)/6.1
    for it = 1:nt
        if (it == 11) GC.enable(false); global t_tic=time(); end      # Start measuring time.
        @parallel loopopt=true step!(
            T2, T, Ci, lam, dt, _dx, _dy, _dz)
        T, T2 = T2, T
    end
    time_s = time() - t_tic

    # Performance
    A_eff = (2*1+1)*1/1e9*nx*ny*nz*sizeof(eltype(T));        # Effective main memory access per iteration [GB] (Lower bound of required memory access: T has to be read and written: 2 whole-array memaccess; Ci has to be read: : 1 whole-array memaccess)
    t_it  = time_s/(nt-10);                                  # Execution time per iteration [s]
    T_eff = A_eff/t_it;                                      # Effective memory throughput [GB/s]
    println("time_s=$time_s t_it=$t_it T_eff=$T_eff");

    nprocs=1; 
    open("./out_$(basename(@__FILE__))_sustained_$(nprocs).txt","a") do io
        println(io, "$(nprocs) $(nx) $(ny) $(nz) $(nt-10) $(time_s) $(A_eff) $(t_it) $(T_eff)")
    end

    # Performance
    A_eff = (2*1+1)*1/1e9*nx*ny*nz*sizeof(eltype(T));        # Effective main memory access per iteration [GB] (Lower bound of required memory access: T has to be read and written: 2 whole-array memaccess; Ci has to be read: : 1 whole-array memaccess)    
    t_it = @belapsed begin @parallel step!($T2, $T, $Ci, $lam, $dt, $_dx, $_dy, $_dz); end
    println("Benchmarktools (min): t_it=$t_it T_eff=$(A_eff/t_it)");

    nprocs=1; nt=11; time_s=t_it; T_eff=A_eff/t_it
    open("./out_$(basename(@__FILE__))_benchmarktools_$(nprocs).txt","a") do io
        println(io, "$(nprocs) $(nx) $(ny) $(nz) $(nt-10) $(time_s) $(A_eff) $(t_it) $(T_eff)")
    end

end

diffusion3D()
