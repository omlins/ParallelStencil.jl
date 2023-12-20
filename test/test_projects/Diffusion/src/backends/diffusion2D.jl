@parallel function diffusion2D_step!(T2, T, Ci, lam, dt, dx, dy)
    @inn(T2) = @inn(T) + dt*(lam*@inn(Ci)*(@d2_xi(T)/dx^2 + @d2_yi(T)/dy^2));
    return
end

function diffusion2D()
# Physics
lam        = 1.0;                                        # Thermal conductivity
cp_min     = 1.0;                                        # Minimal heat capacity
lx, ly     = 10.0, 10.0;                                 # Length of computational domain in dimension x, y and z

# Numerics
nx, ny     = 8, 8;                                       # Number of gridpoints in dimensions x, y and z
nt         = 3;                                          # Number of time steps
dx         = lx/(nx-1);                                  # Space step in x-dimension
dy         = ly/(ny-1);                                  # Space step in y-dimension

# Array initializations
T   = @zeros(nx, ny);
T2  = @zeros(nx, ny);
Ci  = @zeros(nx, ny);

# Initial conditions (heat capacity and temperature with two Gaussian anomalies each)
Ci .= 1.0./( cp_min .+ Data.Array([5*exp(-(((ix-1)*dx-lx/1.5))^2-(((iy-1)*dy-ly/2))^2) +
                                   5*exp(-(((ix-1)*dx-lx/3.0))^2-(((iy-1)*dy-ly/2))^2) for ix=1:size(T,1), iy=1:size(T,2)]) )
T  .= Data.Array([100*exp(-(((ix-1)*dx-lx/2)/2)^2-(((iy-1)*dy-ly/2)/2)^2) +
                   50*exp(-(((ix-1)*dx-lx/2)/2)^2-(((iy-1)*dy-ly/2)/2)^2) for ix=1:size(T,1), iy=1:size(T,2)])
T2 .= T;                                                 # Assign also T2 to get correct boundary conditions.

# Time loop
dt = min(dx^2,dy^2)*cp_min/lam/4.1;                      # Time step for the 2D Heat diffusion
for it = 1:nt
    @parallel diffusion2D_step!(T2, T, Ci, lam, dt, dx, dy);
    T, T2 = T2, T;
end

return Data.Array
end
