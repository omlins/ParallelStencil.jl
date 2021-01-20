const USE_GPU = true  # Use GPU? If this is set false, then no GPU needs to be available
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3)
else
    @init_parallel_stencil(Threads, Float64, 3)
end
using ImplicitGlobalGrid, Plots, Printf, Statistics, LinearAlgebra  # ATTENTION: plotting fails inside plotting library if using flag '--math-mode=fast'.
import MPI
# Global reductions
mean_g(A)    = (mean_l = mean(A);    MPI.Allreduce(mean_l, MPI.SUM, MPI.COMM_WORLD)/MPI.Comm_size(MPI.COMM_WORLD))
maximum_g(A) = (max_l  = maximum(A); MPI.Allreduce(max_l,  MPI.MAX, MPI.COMM_WORLD))
# CPU functions
@views av_zi(A) = (A[2:end-1,2:end-1,2:end-2] .+ A[2:end-1,2:end-1,3:end-1]).*0.5
@views av_za(A) = (A[:,:,1:end-1] .+ A[:,:,2:end]).*0.5
@views inn(A)   =  A[2:end-1,2:end-1,2:end-1]

@parallel function compute_timesteps!(dτVx::Data.Array, dτVy::Data.Array, dτVz::Data.Array, dτPt::Data.Array, Mus::Data.Array, Vsc::Data.Number, Ptsc::Data.Number, min_dxyz2::Data.Number, max_nxyz)
    @all(dτVx) = Vsc*min_dxyz2/@av_xi(Mus)/6.1
    @all(dτVy) = Vsc*min_dxyz2/@av_yi(Mus)/6.1
    @all(dτVz) = Vsc*min_dxyz2/@av_zi(Mus)/6.1
    @all(dτPt) = Ptsc*6.1*@all(Mus)/max_nxyz
    return
end

@parallel function compute_P!(∇V::Data.Array, Pt::Data.Array, Vx::Data.Array, Vy::Data.Array, Vz::Data.Array, dτPt::Data.Array, dx::Data.Number, dy::Data.Number, dz::Data.Number)
    @all(∇V)  = @d_xa(Vx)/dx + @d_ya(Vy)/dy + @d_za(Vz)/dz
    @all(Pt)  = @all(Pt) - @all(dτPt)*@all(∇V)
    return
end

@parallel function compute_τ!(∇V::Data.Array, τxx::Data.Array, τyy::Data.Array, τzz::Data.Array, τxy::Data.Array, τxz::Data.Array, τyz::Data.Array, Vx::Data.Array, Vy::Data.Array, Vz::Data.Array, Mus::Data.Array, dx::Data.Number, dy::Data.Number, dz::Data.Number)
    @all(τxx) = 2.0*@inn_yz(Mus)*(@d_xi(Vx)/dx  - 1.0/3.0*@inn_yz(∇V))
    @all(τyy) = 2.0*@inn_xz(Mus)*(@d_yi(Vy)/dy  - 1.0/3.0*@inn_xz(∇V))
    @all(τzz) = 2.0*@inn_xy(Mus)*(@d_zi(Vz)/dz  - 1.0/3.0*@inn_xy(∇V))
    @all(τxy) = 2.0*@av_xyi(Mus)*(0.5*(@d_yi(Vx)/dy + @d_xi(Vy)/dx))
    @all(τxz) = 2.0*@av_xzi(Mus)*(0.5*(@d_zi(Vx)/dz + @d_xi(Vz)/dx))
    @all(τyz) = 2.0*@av_yzi(Mus)*(0.5*(@d_zi(Vy)/dz + @d_yi(Vz)/dy))
    return
end

@parallel function compute_dV!(Rx::Data.Array, Ry::Data.Array, Rz::Data.Array, dVxdτ::Data.Array, dVydτ::Data.Array, dVzdτ::Data.Array, Pt::Data.Array, Rog::Data.Array, τxx::Data.Array, τyy::Data.Array, τzz::Data.Array, τxy::Data.Array, τxz::Data.Array, τyz::Data.Array, dampX::Data.Number, dampY::Data.Number, dampZ::Data.Number, dx::Data.Number, dy::Data.Number, dz::Data.Number)
    @all(Rx)    = @d_xa(τxx)/dx + @d_ya(τxy)/dy + @d_za(τxz)/dz - @d_xi(Pt)/dx
    @all(Ry)    = @d_ya(τyy)/dy + @d_xa(τxy)/dx + @d_za(τyz)/dz - @d_yi(Pt)/dy
    @all(Rz)    = @d_za(τzz)/dz + @d_xa(τxz)/dx + @d_ya(τyz)/dy - @d_zi(Pt)/dz + @av_zi(Rog)
    @all(dVxdτ) = dampX*@all(dVxdτ) + @all(Rx)
    @all(dVydτ) = dampY*@all(dVydτ) + @all(Ry)
    @all(dVzdτ) = dampZ*@all(dVzdτ) + @all(Rz)
    return
end

@parallel function compute_V!(Vx::Data.Array, Vy::Data.Array, Vz::Data.Array, dVxdτ::Data.Array, dVydτ::Data.Array, dVzdτ::Data.Array, dτVx::Data.Array, dτVy::Data.Array, dτVz::Data.Array)
    @inn(Vx) = @inn(Vx) + @all(dτVx)*@all(dVxdτ)
    @inn(Vy) = @inn(Vy) + @all(dτVy)*@all(dVydτ)
    @inn(Vz) = @inn(Vz) + @all(dτVz)*@all(dVzdτ)
    return
end

@parallel_indices (iy,iz) function bc_x!(A::Data.Array)
    A[  1, iy,  iz] = A[    2,   iy,   iz]
    A[end, iy,  iz] = A[end-1,   iy,   iz]
    return
end

@parallel_indices (ix,iz) function bc_y!(A::Data.Array)
    A[ ix,  1,  iz] = A[   ix,    2,   iz]
    A[ ix,end,  iz] = A[   ix,end-1,   iz]
    return
end

@parallel_indices (ix,iy) function bc_z!(A::Data.Array)
    A[ ix,  iy,  1] = A[   ix,   iy,    2]
    A[ ix,  iy,end] = A[   ix,   iy,end-1]
    return
end

##################################################
@views function Stokes3D()
    # Physics
    lx, ly, lz = 10.0, 10.0, 10.0  # domain extends
    μs0        = 1.0               # matrix viscosity
    μsi        = 0.1               # inclusion viscosity
    ρgi        = 1.0               # inclusion density*gravity perturbation
    # Numerics
    iterMax    = 20000             # maximum number of pseudo-transient iterations
    nout       = 1000              # error checking frequency
    Vdmp       = 4.0               # damping paramter for the momentum equations
    Vsc        = 1.0               # relaxation paramter for the momentum equations pseudo-timesteps limiters
    Ptsc       = 1.0/4.0           # relaxation paramter for the pressure equation pseudo-timestep limiter
    ε          = 1e-6              # nonlinear absolute tolerence
    nx, ny, nz = 127, 127, 127     # numerical grid resolution; should be a mulitple of 32-1 for optimal GPU perf
    # Derived numerics
    me, dims, nprocs, coords, comm = init_global_grid(nx, ny, nz) # MPI initialisation
    select_device()                                               # select one GPU per MPI local rank (if >1 GPU per node)
    dx, dy, dz = lx/(nx_g()-1), ly/(ny_g()-1), lz/(nz_g()-1)      # cell sizes
    min_dxyz2  = min(dx,dy,dz)^2
    max_nxyz   = max(nx_g(),ny_g(),nz_g())
    dampX      = 1.0-Vdmp/nx_g()   # damping term for the x-momentum equation
    dampY      = 1.0-Vdmp/ny_g()   # damping term for the y-momentum equation
    dampZ      = 1.0-Vdmp/nz_g()   # damping term for the z-momentum equation
    # Array allocations
    Pt         = @zeros(nx  ,ny  ,nz  )
    dτPt       = @zeros(nx  ,ny  ,nz  )
    ∇V         = @zeros(nx  ,ny  ,nz  )
    Vx         = @zeros(nx+1,ny  ,nz  )
    Vy         = @zeros(nx  ,ny+1,nz  )
    Vz         = @zeros(nx  ,ny  ,nz+1)
    τxx        = @zeros(nx  ,ny-2,nz-2)
    τyy        = @zeros(nx-2,ny  ,nz-2)
    τzz        = @zeros(nx-2,ny-2,nz  )
    τxy        = @zeros(nx-1,ny-1,nz-2)
    τxz        = @zeros(nx-1,ny-2,nz-1)
    τyz        = @zeros(nx-2,ny-1,nz-1)
    Rx         = @zeros(nx-1,ny-2,nz-2)
    Ry         = @zeros(nx-2,ny-1,nz-2)
    Rz         = @zeros(nx-2,ny-2,nz-1)
    dVxdτ      = @zeros(nx-1,ny-2,nz-2)
    dVydτ      = @zeros(nx-2,ny-1,nz-2)
    dVzdτ      = @zeros(nx-2,ny-2,nz-1)
    dτVx       = @zeros(nx-1,ny-2,nz-2)
    dτVy       = @zeros(nx-2,ny-1,nz-2)
    dτVz       = @zeros(nx-2,ny-2,nz-1)
    # Initial conditions
    Radc       =  zeros(nx  ,ny  ,nz  )
    Rog        =  zeros(nx  ,ny  ,nz  )
    Mus        = μs0*ones(nx,ny,nz)
    Radc      .= [(x_g(ix,dx,Radc)-0.5*lx)^2 + (y_g(iy,dy,Radc)-0.5*ly)^2 + (z_g(iz,dz,Radc)-0.5*lz)^2 for ix=1:size(Radc,1), iy=1:size(Radc,2), iz=1:size(Radc,3)]
    Mus[Radc.<1.0] .= μsi
    Rog[Radc.<1.0] .= ρgi
    Mus        = Data.Array(Mus)
    Rog        = Data.Array(Rog)
    # Preparation of visualisation
    ENV["GKSwstype"]="nul"
    if (me==0)
        if isdir("viz3D_out")==false mkdir("viz3D_out") end; loadpath = "./viz3D_out/"; anim = Animation(loadpath,String[])
        println("Animation directory: $(anim.dir)")
    end
    nx_v, ny_v, nz_v = (nx-2)*dims[1], (ny-2)*dims[2], (nz-2)*dims[3]
    if (nx_v*ny_v*nz_v*sizeof(Data.Number) > 0.8*Sys.free_memory()) error("Not enough memory for visualization.") end
    Pt_v   = zeros(nx_v, ny_v, nz_v) # global array for visu
    Vz_v   = zeros(nx_v, ny_v, nz_v)
    Rz_v   = zeros(nx_v, ny_v, nz_v)
    Pt_inn = zeros(nx-2, ny-2, nz-2) # no halo local array for visu
    Vz_inn = zeros(nx-2, ny-2, nz-2)
    Rz_inn = zeros(nx-2, ny-2, nz-2)
    y_sl2, y_sl = Int(ceil((ny_g()-2)/2)), Int(ceil(ny_g()/2))
    Xi_g, Zi_g  = dx:dx:(lx-dx), dz:dz:(lz-dz) # inner points only
    # Time loop
    @parallel compute_timesteps!(dτVx, dτVy, dτVz, dτPt, Mus, Vsc, Ptsc, min_dxyz2, max_nxyz)
    err=2*ε; iter=1; niter=0; err_evo1=[]; err_evo2=[]
    while err > ε && iter <= iterMax
        if (iter==11)  tic()  end
        @parallel compute_P!(∇V, Pt, Vx, Vy, Vz, dτPt, dx, dy, dz)
        @parallel compute_τ!(∇V, τxx, τyy, τzz, τxy, τxz, τyz, Vx, Vy, Vz, Mus, dx, dy, dz)
        @parallel compute_dV!(Rx, Ry, Rz, dVxdτ, dVydτ, dVzdτ, Pt, Rog, τxx, τyy, τzz, τxy, τxz, τyz, dampX, dampY, dampZ, dx, dy, dz)
        @hide_communication (16, 8, 2) begin # communication/computation overlap
            @parallel compute_V!(Vx, Vy, Vz, dVxdτ, dVydτ, dVzdτ, dτVx, dτVy, dτVz)
            @parallel (1:size(Vy,2), 1:size(Vy,3)) bc_x!(Vy)
            @parallel (1:size(Vz,2), 1:size(Vz,3)) bc_x!(Vz)
            @parallel (1:size(Vx,1), 1:size(Vx,3)) bc_y!(Vx)
            @parallel (1:size(Vz,1), 1:size(Vz,3)) bc_y!(Vz)
            @parallel (1:size(Vx,1), 1:size(Vx,2)) bc_z!(Vx)
            @parallel (1:size(Vy,1), 1:size(Vy,2)) bc_z!(Vy)
            update_halo!(Vx, Vy, Vz)
        end
        if mod(iter,nout)==0
            global mean_Rx, mean_Ry, mean_Rz, mean_∇V
            mean_Rx = mean_g(abs.(Rx)); mean_Ry = mean_g(abs.(Ry)); mean_Rz = mean_g(abs.(Rz)); mean_∇V = mean_g(abs.(∇V))
            err = maximum([mean_Rx, mean_Ry, mean_Rz, mean_∇V])
            push!(err_evo1,maximum([mean_Rx, mean_Ry, mean_Rz, mean_∇V])); push!(err_evo2,iter)
            if (me==0) @printf("Total steps = %d, err = %1.3e [mean_Rx=%1.3e, mean_Ry=%1.3e, mean_Rz=%1.3e, mean_∇V=%1.3e] \n", iter, err, mean_Rx, mean_Ry, mean_Rz, mean_∇V) end
        end
        iter+=1; niter+=1
    end
    # Performance
    wtime    = toc()
    A_eff    = (4*2)/1e9*nx*ny*nz*sizeof(Data.Number)  # Effective main memory access per iteration [GB] (Lower bound of required memory access: Te has to be read and written: 2 whole-array memaccess; Ci has to be read: : 1 whole-array memaccess)
    wtime_it = wtime/(niter-10)                        # Execution time per iteration [s]
    T_eff    = A_eff/wtime_it                          # Effective memory throughput [GB/s]
    if (me==0) @printf("Total steps = %d, err = %1.3e, time = %1.3e sec (@ T_eff = %1.2f GB/s) \n", niter, err, wtime, round(T_eff, sigdigits=2)) end
    # Visualisation
    Pt_inn .= inn(Pt);   gather!(Pt_inn, Pt_v)
    Vz_inn .= av_zi(Vz); gather!(Vz_inn, Vz_v)
    Rz_inn .= av_za(Rz); gather!(Rz_inn, Rz_v)
    if (me==0)
        p1 = heatmap(Xi_g, Zi_g, Pt_v[:,y_sl,:]', aspect_ratio=1, xlims=(Xi_g[1],Xi_g[end]), zlims=(Zi_g[1],Zi_g[end]), c=:inferno, title="Pressure")
        p2 = heatmap(Xi_g, Zi_g, Vz_v[:,y_sl,:]', aspect_ratio=1, xlims=(Xi_g[1],Xi_g[end]), zlims=(Zi_g[1],Zi_g[end]), c=:inferno, title="Vz")
        p4 = heatmap(Xi_g, Zi_g, log10.(abs.(Rz_v[:,y_sl2,:]')), aspect_ratio=1,  xlims=(Xi_g[1],Xi_g[end]), zlims=(Zi_g[1],Zi_g[end]), c=:inferno, title="log10(Rz)")
        p5 = plot(err_evo2,err_evo1, legend=false, xlabel="# iterations", ylabel="log10(error)", linewidth=2, markershape=:circle, markersize=3, labels="max(error)", yaxis=:log10)
        # display(plot(p1, p2, p4, p5))
        plot(p1, p2, p4, p5); frame(anim)
        gif(anim, "Stokes3D.gif", fps = 15)
    end
    finalize_global_grid()
    return
end

Stokes3D()
