using Test
import ParallelStencil: SUPPORTED_PACKAGES, PKG_CUDA, PKG_AMDGPU, PKG_METAL, PKG_KERNELABSTRACTIONS, PKG_POLYESTER, @select_hardware, @current_hardware
import ParallelStencil.ParallelKernel: handle
include(joinpath(@__DIR__, "packages.jl")) # Selection of the packages to test (TEST_PACKAGES), imports of the backend packages.
TEST_PACKAGES = filter!(x->x≠PKG_POLYESTER, TEST_PACKAGES) # NOTE: Polyester is not tested here, because the CPU case is sufficiently covered by the test of the Threads package.
exename = joinpath(Sys.BINDIR, Base.julia_exename())
const TEST_PROJECTS = ["Diffusion3D_minimal"] # ["Diffusion3D_minimal", "Diffusion3D", "Diffusion"]


@testset "$(basename(@__FILE__))" verbose=true begin # NOTE: the package testsets are nested in this testset (and the package loop is run at run time) in order to test all packages even if the tests of one package fail.
for package in TEST_PACKAGES  eval(:(
    @testset "$(basename(@__FILE__)) (package: $(nameof($package)))" begin
        @testset "extensions ($project)" for project in TEST_PROJECTS
            test_script = joinpath(@__DIR__, "test_projects", project, "test", "localtest_diffusion_$(nameof($package)).jl")
            was_success = true
            try
                run(`$exename -O3 --startup-file=no $test_script`)
            catch ex
                was_success = false
            end
            @test was_success
        end;
    end;
)) end
end;
