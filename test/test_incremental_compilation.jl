using Test
import ParallelStencil: SUPPORTED_PACKAGES, PKG_CUDA, PKG_AMDGPU, PKG_METAL, PKG_POLYESTER
TEST_PACKAGES = SUPPORTED_PACKAGES
@static if PKG_CUDA in TEST_PACKAGES
    import CUDA
    if !CUDA.functional() TEST_PACKAGES = filter!(x->x≠PKG_CUDA, TEST_PACKAGES) end
end
@static if PKG_AMDGPU in TEST_PACKAGES
    import AMDGPU
    if !AMDGPU.functional() TEST_PACKAGES = filter!(x->x≠PKG_AMDGPU, TEST_PACKAGES) end
end
@static if PKG_METAL in TEST_PACKAGES
    @static if Sys.isapple()
        import Metal
        if !Metal.functional() TEST_PACKAGES = filter!(x->x≠PKG_METAL, TEST_PACKAGES) end
    else
        TEST_PACKAGES = filter!(x->x≠PKG_METAL, TEST_PACKAGES)
    end
end
@static if PKG_POLYESTER in TEST_PACKAGES
    import Polyester
end
exename = joinpath(Sys.BINDIR, Base.julia_exename())


@static for package in TEST_PACKAGES  eval(:(
    @testset "$(basename(@__FILE__)) (package: $(nameof($package)))" begin
        @testset "incremental compilation" begin
            test_script = joinpath(@__DIR__, "test_projects", "Diffusion3D_$(nameof($package))", "test", "localtest_diffusion.jl")
            was_success = true
            try
                run(`$exename --startup-file=no $test_script`)
            catch ex
                was_success = false
            end
            @test was_success
        end;
    end;
)) end == nothing || true;
