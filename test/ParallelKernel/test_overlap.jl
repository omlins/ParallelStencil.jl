using Test
import ParallelStencil
using ParallelStencil.ParallelKernel
import ParallelStencil.ParallelKernel: @reset_parallel_kernel, @is_initialized, SUPPORTED_PACKAGES, PKG_CUDA, PKG_AMDGPU, PKG_METAL, PKG_POLYESTER
import ParallelStencil.ParallelKernel: @require, @isgpu, is_installed, rmlines
import ParallelStencil.ParallelKernel: checkargs_overlap, overlap_gpu
using ParallelStencil.ParallelKernel.Exceptions

TEST_PACKAGES = SUPPORTED_PACKAGES
@static if PKG_CUDA in TEST_PACKAGES && !is_installed("CUDA")
    TEST_PACKAGES = filter!(x->x≠PKG_CUDA, TEST_PACKAGES)
end
@static if PKG_AMDGPU in TEST_PACKAGES && !is_installed("AMDGPU")
    TEST_PACKAGES = filter!(x->x≠PKG_AMDGPU, TEST_PACKAGES)
end
@static if PKG_METAL in TEST_PACKAGES && !is_installed("Metal")
    TEST_PACKAGES = filter!(x->x≠PKG_METAL, TEST_PACKAGES)
end
@static if PKG_POLYESTER in TEST_PACKAGES && !is_installed("Polyester")
    TEST_PACKAGES = filter!(x->x≠PKG_POLYESTER, TEST_PACKAGES)
end
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
            @testset "@overlap (macro expansion)" begin
                expansion = overlap_gpu(:(begin
                    @parallel (1:sim.Nx, 2:sim.Ny) kernel1!(sim.array1)
                    @parallel (1:sim.Nx, 2:sim.Ny) kernel2!(sim.array2)
                    @parallel (1:sim.Nx, 2:sim.Ny) kernel3!(sim.array3)
                end))
                stmts = filter(x->!isa(x, LineNumberNode), expansion.args)
                expected = [
                    :(@parallel_async (1:sim.Nx, 2:sim.Ny) stream=ParallelStencil.ParallelKernel.@get_stream(1) kernel1!(sim.array1)),
                    :(@parallel_async (1:sim.Nx, 2:sim.Ny) stream=ParallelStencil.ParallelKernel.@get_stream(2) kernel2!(sim.array2)),
                    :(@parallel_async (1:sim.Nx, 2:sim.Ny) stream=ParallelStencil.ParallelKernel.@get_stream(3) kernel3!(sim.array3)),
                    :(ParallelStencil.ParallelKernel.@synchronize(ParallelStencil.ParallelKernel.@get_stream(1))),
                    :(ParallelStencil.ParallelKernel.@synchronize(ParallelStencil.ParallelKernel.@get_stream(2))),
                    :(ParallelStencil.ParallelKernel.@synchronize(ParallelStencil.ParallelKernel.@get_stream(3))),
                ]
                @test length(stmts) == length(expected)
                normalize(ex) = replace(string(rmlines(ex)), r"#= .*? =#\s*" => "") # strip line number annotations
                @test normalize.(stmts) == normalize.(expected)
            end;
            @parallel_indices (ix,iy,iz) function fill_value!(A, value)
                A[ix,iy,iz] = value
                return
            end
            @testset "@overlap execution" begin
                A = @zeros(4, 3, 2)
                B = @zeros(4, 3, 2)
                two = $FloatDefault(2)
                @overlap begin
                    @parallel fill_value!(A, one(eltype(A)))
                    @parallel fill_value!(B, two)
                end
                @test all(Array(A) .== one($FloatDefault))
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
            @test_throws ArgumentError overlap_gpu(:(begin @parallel stream=mystream f(); end))
            @reset_parallel_kernel()
        end;
    end;
))
end
