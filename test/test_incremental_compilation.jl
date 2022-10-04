push!(LOAD_PATH, "@stdlib")  # NOTE: this is needed to enable this test to run from the Pkg manager
push!(LOAD_PATH, joinpath(@__DIR__, ".."))
using Test
using Pkg
import ParallelStencil: SUPPORTED_PACKAGES, PKG_CUDA, PKG_AMDGPU
TEST_PACKAGES = SUPPORTED_PACKAGES
@static if PKG_CUDA in TEST_PACKAGES
    import CUDA
    if !CUDA.functional() TEST_PACKAGES = filter!(x->x≠PKG_CUDA, TEST_PACKAGES) end
end
@static if PKG_AMDGPU in TEST_PACKAGES
    import AMDGPU
    if !AMDGPU.functional() TEST_PACKAGES = filter!(x->x≠PKG_AMDGPU, TEST_PACKAGES) end
end

@static for package in TEST_PACKAGES  eval(:(
    @testset "$(basename(@__FILE__)) (package: $(nameof($package)))" begin
        @testset "incremental compilation" begin
            Pkg.activate(joinpath(@__DIR__, "test_projects", "Diffusion3D_$(nameof($package))"))
            using $(Symbol("Diffusion3D_$package"))
            @test $(Symbol("Diffusion3D_$package")).diffusion3D()
        end;
    end;
)) end == nothing || true;
