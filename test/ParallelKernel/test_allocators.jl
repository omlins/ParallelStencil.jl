using Test
import ParallelStencil
using ParallelStencil.ParallelKernel
import ParallelStencil.ParallelKernel: @reset_parallel_kernel, @is_initialized, SUPPORTED_PACKAGES, PKG_CUDA
import ParallelStencil.ParallelKernel: @require, longnameof
@static if (PKG_CUDA in SUPPORTED_PACKAGES) import CUDA end

@static for package in SUPPORTED_PACKAGES  eval(:(
    @testset "$(basename(@__FILE__)) (package: $(nameof($package)))" begin
        @testset "1. allocator macros" begin
            @require !@is_initialized()
            @init_parallel_kernel($package, Float16)
            @testset "mapping to package" begin
                @require @is_initialized()
                @test @zeros(2,3) == parentmodule($package).zeros(Float16,2,3)
                @test @ones(2,3) == parentmodule($package).ones(Float16,2,3)
                @static if $package == CUDA
                    @test typeof(@rand(2,3)) == typeof(CUDA.CuArray(rand(Float16,2,3)))
                else
                    @test typeof(@rand(2,3)) == typeof(parentmodule($package).rand(Float16,2,3))
                end
            end;
            @reset_parallel_kernel()
        end;
    end;
)) end == nothing || true;
