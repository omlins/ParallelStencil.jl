using ParallelStencil
using ParallelStencil.FiniteDifferences3D
@init_parallel_stencil(CUDA, Float64, 3)

@parallel loopopt=true optvars=T function step!(
    T2, T, Ci, lam, dt, _dx, _dy, _dz)
    @inn(T2) = @inn(T) + dt*(
        lam*@inn(Ci)*(@d2_xi(T)*_dx^2 + 
                      @d2_yi(T)*_dy^2 + 
                      @d2_zi(T)*_dz^2 ) )
    return
end

function diffusion3D()
    # Physics
    lam      = 1.0           #Thermal conductivity
    c0       = 2.0           #Heat capacity
    lx=ly=lz = 1.0           #Domain length x|y|z

    # Numerics
    nx=ny=nz = 512           #Nb gridpoints x|y|z
    nt       = 100           #Nb time steps
    dx       = lx/(nx-1)     #Space step in x
    dy       = ly/(ny-1)     #Space step in y
    dz       = lz/(nz-1)     #Space step in z
    _dx, _dy, _dz = 1.0/dx, 1.0/dy, 1.0/dz

    # Initial conditions
    T  = @ones(nx,ny,nz).*1.7 #Temperature
    T2 = copy(T)              #Temperature (2nd)
    Ci = @ones(nx,ny,nz)./c0  #1/Heat capacity

    # Time loop
    dt = min(dx^2,dy^2,dz^2)/lam/maximum(Ci)/6.1
    for it = 1:nt
        @parallel loopopt=true step!(
            T2, T, Ci, lam, dt, _dx, _dy, _dz)
        T, T2 = T2, T
    end

end

diffusion3D()
