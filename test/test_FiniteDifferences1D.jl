using Test
using ParallelStencil
import ParallelStencil: @reset_parallel_stencil, @is_initialized, SUPPORTED_PACKAGES, PKG_CUDA, PKG_THREADS
import ParallelStencil: @require
using ParallelStencil.FiniteDifferences1D
TEST_PACKAGES = SUPPORTED_PACKAGES
@static if PKG_CUDA in TEST_PACKAGES
    import CUDA
    if !CUDA.functional() TEST_PACKAGES = filter!(x->x≠PKG_CUDA, TEST_PACKAGES) end
end

@static for package in TEST_PACKAGES  eval(:(
    @testset "$(basename(@__FILE__)) (package: $(nameof($package)))" begin
        @init_parallel_stencil($package, Float64, 1)
        @require @is_initialized()
        nx  = 7
        A   =  @rand(nx  );
        Ax  =  @rand(nx+1);
        Axx =  @rand(nx+2);
        R   = @zeros(nx  );
        Rxx = @zeros(nx+2);
        @testset "1. compute macros" begin
            @testset "differences" begin
                @require @is_initialized()
                @parallel d!(R, Ax) = (@all(R) = @d(Ax); return)
                @parallel d2!(R, Axx) = (@all(R) = @d2(Axx); return)
                R.=0; @parallel d!(R, Ax);  @test all(R .== Ax[2:end].-Ax[1:end-1])
                R.=0; @parallel d2!(R, Axx);  @test all(R .== (Axx[3:end].-Axx[2:end-1]).-(Axx[2:end-1].-Axx[1:end-2]))
            end;
            @testset "selection" begin
                @require @is_initialized()
                @parallel all!(R, A) = (@all(R) = @all(A); return)
                @parallel inn!(R, Axx) = (@all(R) = @inn(Axx); return)
                R.=0; @parallel all!(R, A);  @test all(R .== A)
                R.=0; @parallel inn!(R, Axx);  @test all(R .== Axx[2:end-1])
            end;
            @testset "averages" begin
                @require @is_initialized()
                @parallel av!(R, Ax) = (@all(R) = @av(Ax); return)
                R.=0; @parallel av!(R, Ax);  @test all(R .== (Ax[1:end-1].+Ax[2:end]).*0.5)
            end;
            @testset "others" begin
                @require @is_initialized()
                @parallel maxloc!(R, Axx) = (@all(R) = @maxloc(Axx); return)
                R.=0; @parallel maxloc!(R, Axx);  @test all(R .== max.(max.(Axx[3:end],Axx[2:end-1]),Axx[1:end-2]))
            end;
        end;
        @testset "2. apply masks" begin
            @testset "selection" begin
                @require @is_initialized()
                @parallel inn_all!(Rxx, A) = (@inn(Rxx) = @all(A); return)
                @parallel inn_inn!(Rxx, Axx) = (@inn(Rxx) = @inn(Axx); return)
                Rxx.=0; @parallel inn_all!(Rxx, A);  @test all(Rxx[2:end-1] .== A)
                Rxx[2:end-1].=0; @test all(Rxx .== 0)  # Test that boundary values remained zero.
                Rxx.=0; @parallel inn_inn!(Rxx, Axx);  @test all(Rxx[2:end-1] .== Axx[2:end-1])
                Rxx[2:end-1].=0; @test all(Rxx .== 0)  # Test that boundary values remained zero.
            end;
            @testset "differences" begin
                @require @is_initialized()
                @parallel inn_d!(Rxx, Ax) = (@inn(Rxx) = @d(Ax); return)
                @parallel inn_d2!(Rxx, Axx) = (@inn(Rxx) = @d2(Axx); return)
                Rxx.=0; @parallel inn_d!(Rxx, Ax);  @test all(Rxx[2:end-1] .== Ax[2:end].-Ax[1:end-1])
                Rxx[2:end-1].=0; @test all(Rxx .== 0)  # Test that boundary values remained zero.
                Rxx.=0; @parallel inn_d2!(Rxx, Axx);  @test all(Rxx[2:end-1] .== (Axx[3:end].-Axx[2:end-1]).-(Axx[2:end-1].-Axx[1:end-2]))
                Rxx[2:end-1].=0; @test all(Rxx .== 0)  # Test that boundary values remained zero.
            end;
        end;
        @reset_parallel_stencil()
    end;
)) end == nothing || true;
