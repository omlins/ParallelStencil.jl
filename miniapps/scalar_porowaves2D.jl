const USE_GPU = false  # Use GPU? If this is set false, then no GPU needs to be available
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
    macro pow(args...)  esc(:(CUDA.pow($(args...)))) end
else
    @init_parallel_stencil(Threads, Float64, 2)
    pow(x,y) = x^y
    macro pow(args...)  esc(:(pow($(args...)))) end
end
using Plots, Printf, Statistics

@parallel function compute_nonlin!(K_mu::Data.Array, EtaR::Data.Array, Phi::Data.Array, Eta::Data.Array, k_μ0::Data.Number, n::Data.Number)
    @all(K_mu) = k_μ0*@pow(@all(Phi), n)
    @all(EtaR) = @all(Eta)	
    return
end

@parallel_indices (ix,iy) function compute_EtaR!(EtaR::Data.Array, Pe::Data.Array, ηR::Data.Number)
    if (ix<size(EtaR,1) && iy<size(EtaR,2)) if (Pe[ix+1,iy+1]<0.0) EtaR[ix,iy] = ηR; end;  end 
    return
end

@parallel function compute_flux!(qx::Data.Array, qy::Data.Array, K_mu::Data.Array, Pe::Data.Array, Δρg::Data.Number, dx::Data.Number, dy::Data.Number)
    @all(qx) = -@av_xi(K_mu)* @d_xi(Pe)/dx
    @all(qy) = -@av_yi(K_mu)*(@d_yi(Pe)/dy + Δρg)
    return
end

@parallel function compute_Pe_Phi!(dPedt::Data.Array, Pe::Data.Array, Phi::Data.Array, dPhidt::Data.Array, qx::Data.Array, qy::Data.Array, EtaR::Data.Array, β::Data.Number, dt::Data.Number, dx::Data.Number, dy::Data.Number)
    @all(dPedt)  = 1.0/β*(-@d_xa(qx)/dx -@d_ya(qy)/dy -1.0/@all(EtaR)*@inn(Pe))
    @inn(Pe)     = @inn(Pe) + dt*@all(dPedt)
    @all(dPhidt) = -β*@all(dPedt) - 1.0/@all(EtaR)*@inn(Pe)
    @inn(Phi)    = max(0.0, @inn(Phi) + dt*@all(dPhidt))
    return
end

##################################################
@views function poroVEP2D()
    chan      = 0                    # chan=1: channel setup; chan=0: blob setup
    # Physics
    sc        = 1.0+chan*30          # domain extend scaling
    lx, ly    = 250.0/sc, 500.0/sc   # domain extend
    k_μ0      = 1.0                  # permeability/fuild viscosity
    Δρg       = 1.0                  # density*gravity (total-fluid)
    n         = 3.0                  # Karman-Cozeny exponant
    η         = 1.0                  # compaction bulk viscosity
    ηR        = η/(1.0+chan*1000)    # decompaction bulk viscosity
    β         = 0.2                  # bulk compressibility
    λ         = 20.0/sc              # wave length
    ϕ0        = 1.0                  # background porosity
    ϕA        = 2.0                  # perturbation amplitude
    t         = 0.0                  # physical time initialisation
    # Numerics
    nx, ny    = 256, 512             # numerical grid resolution; should be a mulitple of 32 for optimal GPU perf
    nt        = 4e4                  # if chan=0 setup - number of timesteps
    nout      = 5e2                  # if chan=0 setup - plotting frequency
    # nt        = 2e6                  # if chan=1 setup - number of timesteps
    # nout      = 2e4                  # if chan=1 setup - plotting frequency
    # Derived numerics
    dx, dy    = lx/(nx-1), ly/(ny-1) # cell sizes
    # Array allocations
    Eta       = η*@ones(nx-2,ny-2)
    EtaR      =  @zeros(nx-2,ny-2)
    Pe        =  @zeros(nx  ,ny  )
    dPedt     =  @zeros(nx-2,ny-2)
    dPhidt    =  @zeros(nx-2,ny-2)
    K_mu      =  @zeros(nx  ,ny  )
    qx        =  @zeros(nx-1,ny-2)
    qy        =  @zeros(nx-2,ny-1)
    # Initial conditions
    Phi       = ϕ0*ones(nx  ,ny  )
    Radc      = zeros(nx  ,ny  )
    Radc     .= [ (((ix-1)*dx-0.5*lx)/λ/4.)^2 + (((iy-1)*dy-0.2*ly)/λ)^2 for ix=1:size(Radc,1), iy=1:size(Radc,2)]
    Phi[Radc.<1.0] .= Phi[Radc.<1.0] .+ ϕA
    Phi[2:end-1,2:end-1] .= Phi[2:end-1,2:end-1] .+ 1.0/4.1*(diff(diff(Phi[:,2:end-1],dims=1),dims=1).+diff(diff(Phi[2:end-1,:],dims=2),dims=2))
    Phi       = Data.Array(Phi)
    dt        = 1.0
    # Preparation of visualisation
    ENV["GKSwstype"]="nul"; if isdir("viz2D_out")==false mkdir("viz2D_out") end; loadpath = "./viz2D_out/"; anim = Animation(loadpath,String[])
    println("Animation directory: $(anim.dir)")
    X, Y      = -lx/2:dx:lx/2, -ly/2:dy:ly/2
    # Time loop
    for it = 1:nt
        if (it==11)  global wtime0 = Base.time()  end
        @parallel compute_nonlin!(K_mu, EtaR, Phi, Eta, k_μ0, n)
        @parallel compute_EtaR!(EtaR, Pe, ηR)
        if mod(it,100)==1  dt = dx^2/(maximum(K_mu)/β)/5.0  end
        @parallel compute_flux!(qx, qy, K_mu, Pe, Δρg, dx, dy)
        @parallel compute_Pe_Phi!(dPedt, Pe, Phi, dPhidt, qx, qy, EtaR, β, dt, dx, dy)
        t = t + dt
        # Visualisation
        if mod(it,nout)==0
            @printf("it=%d, time=%1.3e \n", it, t)
            p1 = heatmap(X, Y, Array(Phi)', aspect_ratio=1, xlims=(X[1],X[end]), ylims=(Y[1],Y[end]), c=:inferno, title="Porosity")
            p2 = heatmap(X, Y, Array(Pe)',  aspect_ratio=1, xlims=(X[1],X[end]), ylims=(Y[1],Y[end]), c=:inferno, title="Effective pressure")
            # display(plot(p1, p2))
            plot(p1, p2); frame(anim)
        end
    end
    # Performance
    wtime    = Base.time() - wtime0
    A_eff    = (2*2+2)/1e9*nx*ny*sizeof(Data.Number)  # Effective main memory access per iteration [GB] (Lower bound of required memory access: H and dHdτ have to be read and written (dHdτ for damping): 4 whole-array memaccess; B has to be read: 1 whole-array memaccess)
    wtime_it = wtime/(nt-10)                          # Execution time per iteration [s]
    T_eff    = A_eff/wtime_it                         # Effective memory throughput [GB/s]
    @printf("Total steps=%d, time=%1.3e sec (@ T_eff = %1.2f GB/s) \n", nt, wtime, round(T_eff, sigdigits=2))
    return
    gif(anim, "poroVEP2D.gif", fps = 15)
end

poroVEP2D()
