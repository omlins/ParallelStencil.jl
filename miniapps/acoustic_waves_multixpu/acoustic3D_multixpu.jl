const USE_GPU = true  # Use GPU? If this is set false, then no GPU needs to be available
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3)
else
    @init_parallel_stencil(Threads, Float64, 3)
end
using ImplicitGlobalGrid, Plots, Printf, Statistics
import MPI

@parallel function compute_V!(Vx::Data.Array, Vy::Data.Array, Vz::Data.Array, P::Data.Array, dt::Data.Number, ρ::Data.Number, dx::Data.Number, dy::Data.Number, dz::Data.Number)
    @inn(Vx) = @inn(Vx) - dt/ρ*@d_xi(P)/dx
    @inn(Vy) = @inn(Vy) - dt/ρ*@d_yi(P)/dy
    @inn(Vz) = @inn(Vz) - dt/ρ*@d_zi(P)/dz
    return
end

@parallel function compute_P!(P::Data.Array, Vx::Data.Array, Vy::Data.Array, Vz::Data.Array, dt::Data.Number, k::Data.Number, dx::Data.Number, dy::Data.Number, dz::Data.Number)
    @all(P) = @all(P) - dt*k*(@d_xa(Vx)/dx + @d_ya(Vy)/dy + @d_za(Vz)/dz)
    return
end

##################################################
@views function acoustic3D()
    # Physics
    lx, ly, lz = 60.0, 60.0, 60.0  # domain extends
    k          = 1.0               # bulk modulus
    ρ          = 1.0               # density
    t          = 0.0               # physical time
    # Numerics
    nx, ny, nz = 127, 127, 127     # numerical grid resolution; should be a mulitple of 32-1 for optimal GPU perf
    nt         = 1000              # number of timesteps
    nout       = 20                # plotting frequency
    # Derived numerics
    me, dims, nprocs, coords, comm = init_global_grid(nx, ny, nz) # MPI initialisation
    select_device()                                               # select one GPU per MPI local rank (if >1 GPU per node)
    dx, dy, dz = lx/(nx_g()-1), ly/(ny_g()-1), lz/(nz_g()-1)      # cell sizes
    # Array allocations
    P          = @zeros(nx  ,ny  ,nz  )
    Vx         = @zeros(nx+1,ny  ,nz  )
    Vy         = @zeros(nx  ,ny+1,nz  )
    Vz         = @zeros(nx  ,ny  ,nz+1)
    # Initial conditions
    P         .= Data.Array([exp(-(x_g(ix,dx,P)-0.5*lx)^2 -(y_g(iy,dy,P)-0.5*ly)^2 -(z_g(iz,dz,P)-0.5*lz)^2) for ix=1:size(P,1), iy=1:size(P,2), iz=1:size(P,3)])
    dt         = min(dx,dy,dz)/sqrt(k/ρ)/6.1
    # Preparation of visualisation
    ENV["GKSwstype"]="nul"
    if (me==0)
        if isdir("viz3D_out")==false mkdir("viz3D_out") end; loadpath = "./viz3D_out/"; anim = Animation(loadpath,String[])
        println("Animation directory: $(anim.dir)")
    end
    nx_v, ny_v, nz_v = (nx-2)*dims[1], (ny-2)*dims[2], (nz-2)*dims[3]
    if (nx_v*ny_v*nz_v*sizeof(Data.Number) > 0.8*Sys.free_memory()) error("Not enough memory for visualization.") end
    P_v   = zeros(nx_v, ny_v, nz_v) # global array for visu
    P_inn = zeros(nx-2, ny-2, nz-2) # no halo local array for visu
    y_sl  = Int(ceil(ny_g()/2))
    Xi_g  = dx:dx:(lx-dx) # inner points only
    Zi_g  = dz:dz:(lz-dz)
    # Time loop
    for it = 1:nt
        if (it==11) tic() end
        @hide_communication (16, 8, 4) begin # communication/computation overlap
            @parallel compute_V!(Vx, Vy, Vz, P, dt, ρ, dx, dy, dz)
            update_halo!(Vx, Vy, Vz)
        end
        @parallel compute_P!(P, Vx, Vy, Vz, dt, k, dx, dy, dz)
        t = t + dt
        # Visualisation
        if mod(it,nout)==0
            P_inn .= P[2:end-1,2:end-1,2:end-1]; gather!(P_inn, P_v)
            if (me==0) heatmap(Xi_g, Zi_g, P_v[:,y_sl,:]', aspect_ratio=1, xlims=(Xi_g[1],Xi_g[end]), ylims=(Zi_g[1],Zi_g[end]), c=:viridis, title="Pressure"); frame(anim) end
        end
    end
    # Performance
    wtime    = toc()
    A_eff    = (4*2)/1e9*nx*ny*nz*sizeof(Data.Number)  # Effective main memory access per iteration [GB] (Lower bound of required memory access: H and dHdτ have to be read and written (dHdτ for damping): 4 whole-array memaccess; B has to be read: 1 whole-array memaccess)
    wtime_it = wtime/(nt-10)                           # Execution time per iteration [s]
    T_eff    = A_eff/wtime_it                          # Effective memory throughput [GB/s]
    if (me==0) @printf("Total steps=%d, time=%1.3e sec (@ T_eff = %1.2f GB/s) \n", nt, wtime, round(T_eff, sigdigits=2)) end
    if (me==0) gif(anim, "acoustic3D.gif", fps = 15) end
    finalize_global_grid()
    return
end

acoustic3D()
