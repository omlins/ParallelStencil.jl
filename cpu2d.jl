import Pkg; Pkg.activate(".")

const USE_GPU = false
using Test
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2);
else
    @init_parallel_stencil(Threads, Float64, 2);
end

_viscosity(x, y) = exp(1 - ( (1-x) + y*(1-y) ))

@parallel function viscosity!(η, x, y)
    @all(η) = _viscosity(@idx_x(x), @idx_y(y))
    return
end

@parallel function viscosity2!(η, x, y)
    @all(η) = _viscosity(@all(x), @all(y))
    return
end

viscosity(x, y) = [ _viscosity(xi, yi) for xi in x, yi in y]

function testme()

    nx = ny = 64
    x = y = LinRange(0.0, 1.0, nx)
    η1 = @zeros(nx, ny)
    @parallel viscosity!(η1, x, y)
    η2 = [ _viscosity(xi, yi) for xi in x, yi in y]

    return η1 == η2
end

@test  testme() == true

nx, ny = 7, 5
A      =  @rand(nx,   ny  );
R      = @zeros(nx,   ny  );
x = LinRange(zero(Data.Number), one(Data.Number), nx)
y = LinRange(zero(Data.Number), one(Data.Number), ny)
xvec = Data.Array(x)
yvec = Data.Array(y)

@parallel idx_x!(R, A) = (@all(R) = @idx_x(A); return)
@parallel idx_y!(R, A) = (@all(R) = @idx_y(A); return)

R.=0; @parallel idx_x!(R, x); @test reduce(+, Rcol == x for Rcol in eachcol(R)) == size(R,2)
R.=0; @parallel idx_y!(R, y); @test reduce(+, Rrow == y for Rrow in eachrow(R)) == size(R,1)
R.=0; @parallel idx_x!(R, xvec); @test reduce(+, Rcol == xvec for Rcol in eachcol(R)) == size(R,2)
R.=0; @parallel idx_y!(R, yvec); @test reduce(+, Rrow == yvec for Rrow in eachrow(R)) == size(R,1)
        