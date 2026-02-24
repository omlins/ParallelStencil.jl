using Test
import ParallelStencil
using ParallelStencil.ParallelKernel
import ParallelStencil.ParallelKernel: @reset_parallel_kernel, @is_initialized, SUPPORTED_PACKAGES, PKG_CUDA, PKG_AMDGPU, PKG_METAL, PKG_POLYESTER
import ParallelStencil.ParallelKernel: @require, @gorgeousexpand, @isgpu
import ParallelStencil.ParallelKernel: checkargs_overlap, @overlap, overlap_gpu
using ParallelStencil.ParallelKernel.Exceptions
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
Base.retry_load_extensions() # Potentially needed to load the extensions after the packages have been filtered.

for package in TEST_PACKAGES
    FloatDefault = (package == PKG_METAL) ? Float32 : Float64 # Metal does not support Float64
    
eval(:(
    @testset "$(basename(@__FILE__)) (package: $(nameof($package)))" begin
        @testset "1. overlap macro" begin
            @require !@is_initialized()
            @init_parallel_kernel($package, $FloatDefault)
            @require @is_initialized()
            @testset "@overlap block (macro expansion)" begin
                @static if @isgpu($package)
                    block = string(@gorgeousexpand(1, @overlap begin
                        @parallel (1:sim.Nx, 2:sim.Ny) kernel1!(sim.array1)
                        @parallel (1:sim.Nx, 2:sim.Ny) kernel2!(sim.array2)
                        @parallel (1:sim.Nx, 2:sim.Ny) kernel3!(sim.array3)
                    end))
                    @test occursin("@parallel_async (1:sim.Nx, 2:sim.Ny) stream = ParallelStencil.ParallelKernel.@get_stream(1) kernel1!(sim.array1)", block)
                    @test occursin("@parallel_async (1:sim.Nx, 2:sim.Ny) stream = ParallelStencil.ParallelKernel.@get_stream(2) kernel2!(sim.array2)", block)
                    @test occursin("@parallel_async (1:sim.Nx, 2:sim.Ny) stream = ParallelStencil.ParallelKernel.@get_stream(3) kernel3!(sim.array3)", block)
                    @test occursin("ParallelStencil.ParallelKernel.@synchronize ParallelStencil.ParallelKernel.@get_stream(1)", block)
                    @test occursin("ParallelStencil.ParallelKernel.@synchronize ParallelStencil.ParallelKernel.@get_stream(2)", block)
                    @test occursin("ParallelStencil.ParallelKernel.@synchronize ParallelStencil.ParallelKernel.@get_stream(3)", block)
                end;
            end;
            @testset "@overlap execution" begin
                @parallel_indices (ix,iy,iz) function fill_value!(A, value)
                    A[ix,iy,iz] = value
                    return
                end
                A = @zeros(4, 3, 2, eltype=Float32)
                B = @zeros(4, 3, 2, eltype=Float32)
                one = Float32(1)
                two = Float32(2)
                @overlap begin
                    @parallel fill_value!(A, one)
                    @parallel fill_value!(B, two)
                end
                @test all(Array(A) .== one)
                @test all(Array(B) .== two)
            end;
            @reset_parallel_kernel()
        end;
        @testset "2. Exceptions" begin
            @require !@is_initialized()
            @init_parallel_kernel($package, $FloatDefault)
            @require @is_initialized()
            @test_throws ArgumentError checkargs_overlap()
            @test_throws ArgumentError checkargs_overlap(:notablock)
            @test_throws ArgumentError overlap_gpu(:(begin end))
            @test_throws ArgumentError overlap_gpu(:(begin not_parallel_call(); end))
            @test_throws ArgumentError overlap_gpu(:(begin a = 99; end))
            @test_throws ArgumentError overlap_gpu(:(begin @parallel stream=mystream f(); end))
            @reset_parallel_kernel()
        end;
    end;
))
end
