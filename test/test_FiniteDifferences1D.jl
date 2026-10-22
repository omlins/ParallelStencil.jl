using Test
using ParallelStencil
import ParallelStencil: @reset_parallel_stencil, @is_initialized, SUPPORTED_PACKAGES, PKG_CUDA, PKG_AMDGPU, PKG_KERNELABSTRACTIONS, PKG_METAL, PKG_POLYESTER
import ParallelStencil: @require, interpolate
using ParallelStencil.FiniteDifferences1D
using ParallelStencil.FieldAllocators
import ParallelStencil.FieldAllocators: @IField
include(joinpath(@__DIR__, "packages.jl")) # Selection of the packages to test (TEST_PACKAGES), imports of the backend packages.


@testset "$(basename(@__FILE__))" verbose=true begin # NOTE: the package testsets are nested in this testset (and the package loop is run at run time) in order to test all packages even if the tests of one package fail.
for package in TEST_PACKAGES
    FloatDefault = (package == PKG_METAL) ? Float32 : Float64 # Metal does not support Float64

eval(:(
    @testset "$(basename(@__FILE__)) (package: $(nameof($package)))" begin
        $(interpolate(:__padding__, (false, package!=PKG_POLYESTER), :( #TODO: this needs to be restored to (false, true) when Polyester supports padding.
            @testset "(padding=$__padding__)" begin
                @require !@is_initialized()
                @init_parallel_stencil($package, $FloatDefault, 1, padding=__padding__, nonconst_metadata=true)
                @require @is_initialized()
                nx  = (9,)
                A   = @IField(nx,  @rand);
                Ax  = @XField(nx,  @rand);
                Axx =  @Field(nx,  @rand);
                R   = @IField(nx, @zeros);
                Rxx =  @Field(nx, @zeros);
                @testset "1. compute macros" begin
                    @testset "differences" begin
                        @parallel d!(R, Ax) = (@all(R) = @d(Ax); return)
                        @parallel d2!(R, Axx) = (@all(R) = @d2(Axx); return)
                        R.=0; @parallel d!(R, Ax);  @test all(Array(R .== Ax[2:end].-Ax[1:end-1])) # INFO: AMDGPU arrays need to be compared on CPU
                        R.=0; @parallel d2!(R, Axx);  @test all(Array(R .== (Axx[3:end].-Axx[2:end-1]).-(Axx[2:end-1].-Axx[1:end-2])))
                    end;
                    @testset "selection" begin
                        @parallel all!(R, A) = (@all(R) = @all(A); return)
                        @parallel inn!(R, Axx) = (@all(R) = @inn(Axx); return)
                        R.=0; @parallel all!(R, A);  @test all(Array(R .== A))
                        R.=0; @parallel inn!(R, Axx);  @test all(Array(R .== Axx[2:end-1]))
                    end;
                    @testset "averages" begin
                        @parallel av!(R, Ax) = (@all(R) = @av(Ax); return)
                        R.=0; @parallel av!(R, Ax);  @test all(Array(R .== (Ax[1:end-1].+Ax[2:end]).*$FloatDefault(0.5)))
                    end;
                    @testset "harmonic averages" begin
                        @parallel harm!(R, Ax) = (@all(R) = @harm(Ax); return)
                        R.=0; @parallel harm!(R, Ax);  @test all(Array(R .== 2 ./(1 ./Ax[1:end-1].+1 ./Ax[2:end])))
                    end;
                    @testset "others" begin
                        @parallel maxloc!(R, Axx) = (@all(R) = @maxloc(Axx); return)
                        R.=0; @parallel maxloc!(R, Axx);  @test all(Array(R .== max.(max.(Axx[3:end],Axx[2:end-1]),Axx[1:end-2])))
                    end;
                end;
                @testset "2. apply masks" begin
                    @testset "selection" begin
                        @parallel inn_all!(Rxx, A) = (@inn(Rxx) = @all(A); return)
                        @parallel inn_inn!(Rxx, Axx) = (@inn(Rxx) = @inn(Axx); return)
                        Rxx.=0; @parallel inn_all!(Rxx, A);  @test all(Array(Rxx[2:end-1] .== A))
                        Rxx[2:end-1].=0; @test all(Array(Rxx .== 0))  # Test that boundary values remained zero.
                        Rxx.=0; @parallel inn_inn!(Rxx, Axx);  @test all(Array(Rxx[2:end-1] .== Axx[2:end-1]))
                        Rxx[2:end-1].=0; @test all(Array(Rxx .== 0))  # Test that boundary values remained zero.
                    end;
                    @testset "differences" begin
                        @parallel inn_d!(Rxx, Ax) = (@inn(Rxx) = @d(Ax); return)
                        @parallel inn_d2!(Rxx, Axx) = (@inn(Rxx) = @d2(Axx); return)
                        Rxx.=0; @parallel inn_d!(Rxx, Ax);  @test all(Array(Rxx[2:end-1] .== Ax[2:end].-Ax[1:end-1]))
                        Rxx[2:end-1].=0; @test all(Array(Rxx .== 0))  # Test that boundary values remained zero.
                        Rxx.=0; @parallel inn_d2!(Rxx, Axx);  @test all(Array(Rxx[2:end-1] .== (Axx[3:end].-Axx[2:end-1]).-(Axx[2:end-1].-Axx[1:end-2])))
                        Rxx[2:end-1].=0; @test all(Array(Rxx .== 0))  # Test that boundary values remained zero.
                    end;
                end;
                @reset_parallel_stencil()
            end;
        )))
    end;
))

end
end;
