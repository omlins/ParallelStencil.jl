using Test
import ParallelStencil: SUPPORTED_PACKAGES, PKG_CUDA, PKG_AMDGPU
TEST_PACKAGES = SUPPORTED_PACKAGES
@static if PKG_CUDA in TEST_PACKAGES
    import ParallelStencil.CUDA
    if !CUDA.functional() TEST_PACKAGES = filter!(x->x≠PKG_CUDA, TEST_PACKAGES) end
end
@static if PKG_AMDGPU in TEST_PACKAGES
    import ParallelStencil.AMDGPU
    if !AMDGPU.functional() TEST_PACKAGES = filter!(x->x≠PKG_AMDGPU, TEST_PACKAGES) end
end
exename = joinpath(Sys.BINDIR, Base.julia_exename())

@static for package in TEST_PACKAGES  eval(:(
    @testset "$(basename(@__FILE__)) (package: $(nameof($package)))" begin
        @testset "extensions" begin
            test_script = joinpath(@__DIR__, "test_projects", "Diffusion3D", "test", "localtest_diffusion_$(nameof($package)).jl")
            was_success = true
            try
                run(`$exename -O3 --startup-file=no --check-bounds=no $test_script`)
            catch ex
                was_success = false
            end
            @test was_success
        end;
    end;
)) end == nothing || true;