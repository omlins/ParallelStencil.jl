push!(LOAD_PATH, joinpath(@__DIR__, ".."))
using Test
using Pkg
import ParallelStencil: SUPPORTED_PACKAGES, PKG_CUDA
@static if (PKG_CUDA in SUPPORTED_PACKAGES) import CUDA end

@static for package in SUPPORTED_PACKAGES  eval(:(
    @testset "$(basename(@__FILE__)) (package: $(nameof($package)))" begin
        @testset "incremental compilation" begin
            Pkg.activate(joinpath(@__DIR__, "test_projects", "Diffusion3D_$(nameof($package))"))
            using $(Symbol("Diffusion3D_$package"))
            @test $(Symbol("Diffusion3D_$package")).diffusion3D()
        end;
    end;
)) end == nothing || true;
