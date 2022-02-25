import Pkg; Pkg.activate(".")

const USE_GPU = false
using Test
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3);
else
    @init_parallel_stencil(Threads, Float64, 3);
end

_viscosity(x, y, z) = exp(1 - ( (1-x) + y*(1-y) + z*(1-z)))

@parallel function viscosity!(η, x, y, z)
    @all(η) = _viscosity(@idx_x(x), @idx_y(y), @idx_z(z))
    return
end

function testme()

    nx = ny = nz = 8
    x = y = z = LinRange(0.0, 1.0, nx)
    η1 = @zeros(nx, ny, nz)
    @parallel viscosity!(η1, x, y, z)
    η2 = [ _viscosity(xi, yi, zi) for xi in x, yi in y, zi in z]

    return η1 == η2
end

@test  testme() == true

nx, ny, nz = 7, 5, 6
A       =  @rand(nx  , ny  , nz  );
R       = @zeros(nx  , ny  , nz  );
x = LinRange(zero(Data.Number), one(Data.Number), nx)
y = LinRange(zero(Data.Number), one(Data.Number), ny)
z = LinRange(zero(Data.Number), one(Data.Number), nz)
xvec = Data.Array(x)
yvec = Data.Array(y)
zvec = Data.Array(z)

@parallel idx_x!(R, A) = (@all(R) = @idx_x(A); return)
@parallel idx_y!(R, A) = (@all(R) = @idx_y(A); return)
@parallel idx_z!(R, A) = (@all(R) = @idx_z(A); return)

R.=0; @parallel idx_x!(R, x); @test reduce(+, Rcol == x for Rcol in eachcol(collect(eachslice(R, dims=3))[1])) == size(R,2)
R.=0; @parallel idx_y!(R, y); @test reduce(+, Rrow == y for Rrow in eachrow(collect(eachslice(R, dims=3))[1])) == size(R,1)
R.=0; @parallel idx_z!(R, z); @test reduce(+, Rrow == z for Rrow in eachrow(collect(eachslice(R, dims=1))[1])) == size(R,2)

R.=0; @parallel idx_x!(R, xvec); @test reduce(+, Rcol == x for Rcol in eachcol(collect(eachslice(R, dims=3))[1])) == size(R,2)
R.=0; @parallel idx_y!(R, yvec); @test reduce(+, Rrow == y for Rrow in eachrow(collect(eachslice(R, dims=3))[1])) == size(R,1)
R.=0; @parallel idx_z!(R, zvec); @test reduce(+, Rrow == z for Rrow in eachrow(collect(eachslice(R, dims=1))[1])) == size(R,2)