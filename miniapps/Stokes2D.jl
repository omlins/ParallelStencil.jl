const USE_GPU = false  # Use GPU? If this is set false, then no GPU needs to be available
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end
using Plots, Printf, Statistics, LinearAlgebra

@parallel function compute_timesteps!(dτVx::Data.Array, dτVy::Data.Array, dτPt::Data.Array, Mus::Data.Array, Vsc::Data.Number, Ptsc::Data.Number, min_dxy2::Data.Number, max_nxy::Int)
    @all(dτVx) = Vsc*min_dxy2/@av_xi(Mus)/4.1
    @all(dτVy) = Vsc*min_dxy2/@av_yi(Mus)/4.1
    @all(dτPt) = Ptsc*4.1*@all(Mus)/max_nxy
    return
end

@parallel function compute_P!(∇V::Data.Array, Pt::Data.Array, Vx::Data.Array, Vy::Data.Array, dτPt::Data.Array, dx::Data.Number, dy::Data.Number)
    @all(∇V)  = @d_xa(Vx)/dx + @d_ya(Vy)/dy
    @all(Pt)  = @all(Pt) - @all(dτPt)*@all(∇V)
    return
end

@parallel function compute_τ!(∇V::Data.Array, τxx::Data.Array, τyy::Data.Array, τxy::Data.Array, Vx::Data.Array, Vy::Data.Array, Mus::Data.Array, dx::Data.Number, dy::Data.Number)
    @all(τxx) = 2.0*@all(Mus)*(@d_xa(Vx)/dx - 1.0/3.0*@all(∇V))
    @all(τyy) = 2.0*@all(Mus)*(@d_ya(Vy)/dy - 1.0/3.0*@all(∇V))
    @all(τxy) = 2.0*@av(Mus)*(0.5*(@d_yi(Vx)/dy + @d_xi(Vy)/dx))
    return
end

@parallel function compute_dV!(Rx::Data.Array, Ry::Data.Array, dVxdτ::Data.Array, dVydτ::Data.Array, Pt::Data.Array, Rog::Data.Array, τxx::Data.Array, τyy::Data.Array, τxy::Data.Array, dampX::Data.Number, dampY::Data.Number, dx::Data.Number, dy::Data.Number)
    @all(Rx)    = @d_xi(τxx)/dx + @d_ya(τxy)/dy - @d_xi(Pt)/dx
    @all(Ry)    = @d_yi(τyy)/dy + @d_xa(τxy)/dx - @d_yi(Pt)/dy + @av_yi(Rog)
    @all(dVxdτ) = dampX*@all(dVxdτ) + @all(Rx)
    @all(dVydτ) = dampY*@all(dVydτ) + @all(Ry)
    return
end

@parallel function compute_V!(Vx::Data.Array, Vy::Data.Array, dVxdτ::Data.Array, dVydτ::Data.Array, dτVx::Data.Array, dτVy::Data.Array)
    @inn(Vx) = @inn(Vx) + @all(dτVx)*@all(dVxdτ)
    @inn(Vy) = @inn(Vy) + @all(dτVy)*@all(dVydτ)
    return
end

@parallel_indices (ix,iy) function bc_x!(A::Data.Array)
    A[1  , iy] = A[2    , iy]
    A[end, iy] = A[end-1, iy]
    return
end

@parallel_indices (ix,iy) function bc_y!(A::Data.Array)
    A[ix, 1  ] = A[ix, 2    ]
    A[ix, end] = A[ix, end-1]
    return
end

##################################################
@views function Stokes2D()
    # Physics
    lx, ly    = 10.0, 10.0  # domain extends
    μs0       = 1.0         # matrix viscosity
    μsi       = 0.1         # inclusion viscosity
    ρgi       = 1.0         # inclusion density*gravity perturbation
    # Numerics
    iterMax   = 10000       # maximum number of pseudo-transient iterations
    nout      = 200         # error checking frequency
    Vdmp      = 4.0         # damping paramter for the momentum equations
    Vsc       = 1.0         # relaxation paramter for the momentum equations pseudo-timesteps limiters
    Ptsc      = 1.0/4.0     # relaxation paramter for the pressure equation pseudo-timestep limiter
    ε         = 1e-6        # nonlinear absolute tolerence
    nx, ny    = 127, 127    # numerical grid resolution; should be a mulitple of 32-1 for optimal GPU perf
    # Derived numerics
    dx, dy    = lx/(nx-1), ly/(ny-1) # cell sizes
    min_dxy2  = min(dx,dy)^2
    max_nxy   = max(nx,ny)
    dampX     = 1.0-Vdmp/nx # damping term for the x-momentum equation
    dampY     = 1.0-Vdmp/ny # damping term for the y-momentum equation
    # Array allocations
    Pt        = @zeros(nx  ,ny  )
    dτPt      = @zeros(nx  ,ny  )
    ∇V        = @zeros(nx  ,ny  )
    Vx        = @zeros(nx+1,ny  )
    Vy        = @zeros(nx  ,ny+1)
    τxx       = @zeros(nx  ,ny  )
    τyy       = @zeros(nx  ,ny  )
    τxy       = @zeros(nx-1,ny-1)
    Rx        = @zeros(nx-1,ny-2)
    Ry        = @zeros(nx-2,ny-1)
    dVxdτ     = @zeros(nx-1,ny-2)
    dVydτ     = @zeros(nx-2,ny-1)
    dτVx      = @zeros(nx-1,ny-2)
    dτVy      = @zeros(nx-2,ny-1)
    # Initial conditions
    Radc      =  zeros(nx  ,ny  )
    Rog       =  zeros(nx  ,ny  )
    Mus       = μs0*ones(nx,ny)
    Radc     .= [((ix-1)*dx-0.5*lx)^2 + ((iy-1)*dy-0.5*ly)^2 for ix=1:size(Radc,1), iy=1:size(Radc,2)]
    Mus[Radc.<1.0] .= μsi
    Rog[Radc.<1.0] .= ρgi
    Mus       = Data.Array(Mus)
    Rog       = Data.Array(Rog)
    # Preparation of visualisation
    ENV["GKSwstype"]="nul"; if isdir("viz2D_out")==false mkdir("viz2D_out") end; loadpath = "./viz2D_out/"; anim = Animation(loadpath,String[])
    println("Animation directory: $(anim.dir)")
    X, Y, Yv  = 0:dx:lx, 0:dy:ly, (-dy/2):dy:(ly+dy/2)
    # Time loop
    @parallel compute_timesteps!(dτVx, dτVy, dτPt, Mus, Vsc, Ptsc, min_dxy2, max_nxy)
    err=2*ε; iter=1; niter=0; err_evo1=[]; err_evo2=[]
    while err > ε && iter <= iterMax
        if (iter==11)  global wtime0 = Base.time()  end
        @parallel compute_P!(∇V, Pt, Vx, Vy, dτPt, dx, dy)
        @parallel compute_τ!(∇V, τxx, τyy, τxy, Vx, Vy, Mus, dx, dy)
        @parallel compute_dV!(Rx, Ry, dVxdτ, dVydτ, Pt, Rog, τxx, τyy, τxy, dampX, dampY, dx, dy)
        @parallel compute_V!(Vx, Vy, dVxdτ, dVydτ, dτVx, dτVy)
        @parallel (1:size(Vx,1), 1:size(Vx,2)) bc_y!(Vx)
        @parallel (1:size(Vy,1), 1:size(Vy,2)) bc_x!(Vy)
        if mod(iter,nout)==0
            global mean_Rx, mean_Ry, mean_∇V
            mean_Rx = mean(abs.(Rx)); mean_Ry = mean(abs.(Ry)); mean_∇V = mean(abs.(∇V))
            err = maximum([mean_Rx, mean_Ry, mean_∇V])
            push!(err_evo1, maximum([mean_Rx, mean_Ry, mean_∇V])); push!(err_evo2,iter)
            @printf("Total steps = %d, err = %1.3e [mean_Rx=%1.3e, mean_Ry=%1.3e, mean_∇V=%1.3e] \n", iter, err, mean_Rx, mean_Ry, mean_∇V)
        end
        iter+=1; niter+=1
    end
    # Performance
    wtime    = Base.time() - wtime0
    A_eff    = (3*2)/1e9*nx*ny*sizeof(Data.Number)  # Effective main memory access per iteration [GB] (Lower bound of required memory access: Te has to be read and written: 2 whole-array memaccess; Ci has to be read: : 1 whole-array memaccess)
    wtime_it = wtime/(niter-10)                     # Execution time per iteration [s]
    T_eff    = A_eff/wtime_it                       # Effective memory throughput [GB/s]
    @printf("Total steps = %d, err = %1.3e, time = %1.3e sec (@ T_eff = %1.2f GB/s) \n", niter, err, wtime, round(T_eff, sigdigits=2))
    # Visualisation
    p1 = heatmap(X,  Y, Array(Pt)', aspect_ratio=1, xlims=(X[1],X[end]), ylims=(Y[1],Y[end]), c=:inferno, title="Pressure")
    p2 = heatmap(X, Yv, Array(Vy)', aspect_ratio=1, xlims=(X[1],X[end]), ylims=(Yv[1],Yv[end]), c=:inferno, title="Vy")
    p4 = heatmap(X[2:end-1], Yv[2:end-1], log10.(abs.(Array(Ry)')), aspect_ratio=1, xlims=(X[2],X[end-1]), ylims=(Yv[2],Yv[end-1]), c=:inferno, title="log10(Ry)")
    p5 = plot(err_evo2,err_evo1, legend=false, xlabel="# iterations", ylabel="log10(error)", linewidth=2, markershape=:circle, markersize=3, labels="max(error)", yaxis=:log10)
    # display(plot(p1, p2, p4, p5))
    plot(p1, p2, p4, p5); frame(anim)
    gif(anim, "Stokes2D.gif", fps = 15)
    return
end

Stokes2D()
