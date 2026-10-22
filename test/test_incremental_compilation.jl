using Test
import ParallelStencil: SUPPORTED_PACKAGES, PKG_CUDA, PKG_AMDGPU, PKG_METAL, PKG_KERNELABSTRACTIONS, PKG_POLYESTER, @select_hardware, @current_hardware
include(joinpath(@__DIR__, "packages.jl")) # Selection of the packages to test (TEST_PACKAGES), imports of the backend packages.
exename = joinpath(Sys.BINDIR, Base.julia_exename())


@testset "$(basename(@__FILE__))" verbose=true begin # NOTE: the package testsets are nested in this testset (and the package loop is run at run time) in order to test all packages even if the tests of one package fail.
for package in TEST_PACKAGES  eval(:(
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
)) end
end;
