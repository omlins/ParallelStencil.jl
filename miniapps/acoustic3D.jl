const USE_GPU = false  # Use GPU? If this is set false, then no GPU needs to be available
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3)
else
    @init_parallel_stencil(Threads, Float64, 3)
end
using Plots, Printf, Statistics

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
    lx, ly, lz = 40.0, 40.0, 40.0  # domain extends
    k          = 1.0               # bulk modulus
    ρ          = 1.0               # density
    t          = 0.0               # physical time
    # Numerics
    nx, ny, nz = 255, 255, 255     # numerical grid resolution; should be a mulitple of 32-1 for optimal GPU perf
    nt         = 1000              # number of timesteps
    nout       = 10                # plotting frequency
    # Derived numerics
    dx, dy, dz = lx/(nx-1), ly/(ny-1), lz/(nz-1) # cell sizes
    # Array allocations
    P          = @zeros(nx  ,ny  ,nz  )
    Vx         = @zeros(nx+1,ny  ,nz  )
    Vy         = @zeros(nx  ,ny+1,nz  )
    Vz         = @zeros(nx  ,ny  ,nz+1)
    # Initial conditions
    P         .= Data.Array([exp(-((ix-1)*dx-0.5*lx)^2 -((iy-1)*dy-0.5*ly)^2 -((iz-1)*dz-0.5*lz)^2) for ix=1:size(P,1), iy=1:size(P,2), iz=1:size(P,3)])
    dt         = min(dx,dy,dz)/sqrt(k/ρ)/6.1
    # Preparation of visualisation
    ENV["GKSwstype"]="nul"; if isdir("viz3D_out")==false mkdir("viz3D_out") end; loadpath = "./viz3D_out/"; anim = Animation(loadpath,String[])
    println("Animation directory: $(anim.dir)")
    y_sl       = Int(ceil(ny/2))
    X, Y, Z    = -lx/2:dx:lx/2, -ly/2:dy:ly/2, -lz/2:dz:lz/2
    # Time loop
    for it = 1:nt
        if (it==11)  global wtime0 = Base.time()  end
        @parallel compute_V!(Vx, Vy, Vz, P, dt, ρ, dx, dy, dz)
        @parallel compute_P!(P, Vx, Vy, Vz, dt, k, dx, dy, dz)
        t = t + dt
        # Visualisation
        if mod(it,nout)==0
            heatmap(X, Z, Array(P)[:,y_sl,:]', aspect_ratio=1, xlims=(X[1],X[end]), ylims=(Z[1],Z[end]), c=:viridis, title="Pressure"); frame(anim)
        end
    end
    # Performance
    wtime    = Base.time() - wtime0
    A_eff    = (4*2)/1e9*nx*ny*nz*sizeof(Data.Number)  # Effective main memory access per iteration [GB] (Lower bound of required memory access: H and dHdτ have to be read and written (dHdτ for damping): 4 whole-array memaccess; B has to be read: 1 whole-array memaccess)
    wtime_it = wtime/(nt-10)                           # Execution time per iteration [s]
    T_eff    = A_eff/wtime_it                          # Effective memory throughput [GB/s]
    @printf("Total steps=%d, time=%1.3e sec (@ T_eff = %1.2f GB/s) \n", nt, wtime, round(T_eff, sigdigits=2))
    gif(anim, "acoustic3D.gif", fps = 15)
    return
end

acoustic3D()
