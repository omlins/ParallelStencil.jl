const USE_GPU = false  # Use GPU? If this is set false, then no GPU needs to be available
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end
using Plots, Printf, Statistics

#=
This variation of the acoustic2D miniapp implements a perfectly-matched-layer as described in:

Treeby, Bradley E., and Benjamin T. Cox. "k-Wave: MATLAB toolbox for the simulation 
and reconstruction of photoacoustic wave fields." Journal of biomedical optics 
15.2 (2010): 021314-021314.

Cox, Benjamin T., et al. "k-space propagation models for acoustically heterogeneous media: 
Application to biomedical photoacoustics." The Journal of the Acoustical Society of America 
121.6 (2007): 3453-3464.

and the k-wave user manual: http://www.k-wave.org/manual/k-wave_user_manual_1.1.pdf
=#

@parallel function compute_V!(Vx::Data.Array, Vy::Data.Array, P::Data.Array, αx::Data.Array, αy::Data.Array, dt::Data.Number, ρ::Data.Number, dx::Data.Number, dy::Data.Number)
    @inn(Vx) = @inn(Vx) - dt*(1/ρ*@d_xi(P)/dx + @inn(αx)*@inn(Vx))
    @inn(Vy) = @inn(Vy) - dt*(1/ρ*@d_yi(P)/dy + @inn(αy)*@inn(Vy))
    return
end

@parallel function compute_P!(P::Data.Array, Vx::Data.Array, Vy::Data.Array, αx::Data.Array, αy::Data.Array, dt::Data.Number, ρ::Data.Number, dx::Data.Number, dy::Data.Number)
    @all(P) = @all(P) - dt*(ρ*(@d_xa(Vx)/dx + @d_ya(Vy)/dy)+(@all(αx) + @all(αy))*@all(P)) 
    return
end

##################################################
@views function acoustic2D()
    # Physics
    lx, ly    = 40.0, 40.0  # domain extends
    k         = 1.0         # bulk modulus
    ρ         = 1.0         # density
    t         = 0.0         # physical time
    # Numerics
    nx, ny    = 255, 255    # numerical grid resolution; should be a mulitple of 32-1 for optimal GPU perf
    nt        = 1000        # number of timesteps
    nout      = 10          # plotting frequency
    # Derived numerics
    dx, dy    = lx/(nx-1), ly/(ny-1) # cell sizes
    # Array allocations
    P         = @zeros(nx  ,ny  )
    Vx        = @zeros(nx+1,ny  )
    Vy        = @zeros(nx  ,ny+1)
    αx        = @zeros(nx, ny)
    αy        = @zeros(nx, ny)

    # Initial conditions
    P        .= Data.Array([exp(-((ix-1)*dx-0.5*lx)^2 -((iy-1)*dy-0.5*ly)^2) for ix=1:size(P,1), iy=1:size(P,2)])
    dt        = min(dx,dy)/sqrt(k/ρ)/4.1
    
    c       = √(k/ρ) # Sound speed
    αmax    = 4*c/dx
    ξ0      = 1
    ξmax    = 32
    m       = 4
    ξ       = ξ0:ξmax
    α       = @. αmax * ((ξ-ξ0)/(ξmax-ξ0))^m # Equation (2.28) from the k-wave user manual.

    αx[:, 1:ξmax]           .= reverse(α')
    αx[:, end-ξmax+1:end]   .= α'
    αy[1:ξmax,:]            .= reverse(α)
    αy[end-ξmax+1:end,:]    .= α

    αx = Data.Array(αx)
    αy = Data.Array(αy)

    # Preparation of visualisation
    ENV["GKSwstype"]="nul"; if isdir("viz2D_out")==false mkdir("viz2D_out") end; loadpath = "./viz2D_out/"; anim = Animation(loadpath,String[])
    println("Animation directory: $(anim.dir)")
    X, Y      = -lx/2:dx:lx/2, -ly/2:dy:ly/2
    # Time loop
    for it = 1:nt
        if (it==11)  global wtime0 = Base.time()  end
        @parallel compute_V!(Vx, Vy, P, αx, αy, dt, ρ, dx, dy)
        @parallel compute_P!(P, Vx, Vy, αx, αy, dt, k, dx, dy)
        t = t + dt
        # Visualisation
        if mod(it,nout)==0
            heatmap(X, Y, Array(P)', aspect_ratio=1, xlims=(X[1],X[end]), ylims=(Y[1],Y[end]), c=:viridis, title="Pressure", clims = (-0.5, 0.5)); frame(anim)
        end
    end
    # Performance
    wtime    = Base.time()-wtime0
    A_eff    = (3*2)/1e9*nx*ny*sizeof(Data.Number)  # Effective main memory access per iteration [GB] (Lower bound of required memory access: H and dHdτ have to be read and written (dHdτ for damping): 4 whole-array memaccess; B has to be read: 1 whole-array memaccess)
    wtime_it = wtime/(nt-10)                        # Execution time per iteration [s]
    T_eff    = A_eff/wtime_it                       # Effective memory throughput [GB/s]
    @printf("Total steps=%d, time=%1.3e sec (@ T_eff = %1.2f GB/s) \n", nt, wtime, round(T_eff, sigdigits=2))
    gif(anim, "acoustic2D.gif", fps = 15)
    return
end

acoustic2D()