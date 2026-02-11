using Test
import ParallelStencil: SUPPORTED_PACKAGES, PKG_CUDA, PKG_AMDGPU, PKG_METAL, PKG_KERNELABSTRACTIONS, PKG_POLYESTER, @select_hardware, @current_hardware
import ParallelStencil.ParallelKernel: handle
TEST_PACKAGES = SUPPORTED_PACKAGES
TEST_PACKAGES = filter!(x->x≠PKG_POLYESTER, TEST_PACKAGES) # NOTE: Polyester is not tested here, because the CPU case is sufficiently covered by the test of the Threads package.
@static if PKG_CUDA in TEST_PACKAGES
    import CUDA
    if !CUDA.functional() TEST_PACKAGES = filter!(x->x≠PKG_CUDA, TEST_PACKAGES) end
end
@static if PKG_AMDGPU in TEST_PACKAGES
    import AMDGPU
    if !AMDGPU.functional() TEST_PACKAGES = filter!(x->x≠PKG_AMDGPU, TEST_PACKAGES) end
end
@static if PKG_KERNELABSTRACTIONS in TEST_PACKAGES
    import KernelAbstractions
    if !KernelAbstractions.functional(KernelAbstractions.CPU()) TEST_PACKAGES = filter!(x->x≠PKG_KERNELABSTRACTIONS, TEST_PACKAGES) end
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
const TEST_PROJECTS = ["Diffusion3D_minimal"] # ["Diffusion3D_minimal", "Diffusion3D", "Diffusion"]


@static for package in TEST_PACKAGES  eval(:(
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
)) end == nothing || true;
