using Test
import ParallelStencil
using ParallelStencil.ParallelKernel
import ParallelStencil.ParallelKernel: @reset_parallel_kernel, @is_initialized, @get_numbertype, NUMBERTYPE_NONE, SUPPORTED_PACKAGES, PKG_CUDA
import ParallelStencil.ParallelKernel: @require
TEST_PACKAGES = SUPPORTED_PACKAGES
@static if PKG_CUDA in TEST_PACKAGES
    import CUDA
    if !CUDA.functional() TEST_PACKAGES = filter!(x->x≠PKG_CUDA, TEST_PACKAGES) end
end

@static for package in TEST_PACKAGES  eval(:(
    @testset "$(basename(@__FILE__)) (package: $(nameof($package)))" begin
        @testset "1. allocator macros (with default numbertype)" begin
            @require !@is_initialized()
            @init_parallel_kernel($package, Float16)
            @require @is_initialized()
            @testset "mapping to package" begin
                @test @zeros(2,3)                == parentmodule($package).zeros(Float16,2,3)
                @test @zeros(2,3,eltype=Float32) == parentmodule($package).zeros(Float32,2,3)
                @test @ones(2,3)                 == parentmodule($package).ones(Float16,2,3)
                @test @ones(2,3,eltype=Float32)  == parentmodule($package).ones(Float32,2,3)
                @static if $package == $PKG_CUDA
                    @test typeof(@rand(2,3))                == typeof(CUDA.CuArray(rand(Float16,2,3)))
                    @test typeof(@rand(2,3,eltype=Float64)) == typeof(CUDA.CuArray(rand(Float64,2,3)))
                else
                    @test typeof(@rand(2,3))                == typeof(parentmodule($package).rand(Float16,2,3))
                    @test typeof(@rand(2,3,eltype=Float64)) == typeof(parentmodule($package).rand(Float64,2,3))
                end
            end;
            @reset_parallel_kernel()
        end;
        @testset "2. allocator macros (no default numbertype)" begin
            @require !@is_initialized()
            @init_parallel_kernel(package = $package)
            @require @is_initialized()
            @require @get_numbertype() == NUMBERTYPE_NONE
            @testset "mapping to package" begin
                @test @zeros(2,3,eltype=Float32) == parentmodule($package).zeros(Float32,2,3)
                @test @ones(2,3,eltype=Float32)  == parentmodule($package).ones(Float32,2,3)
                @static if $package == $PKG_CUDA
                    @test typeof(@rand(2,3,eltype=Float64)) == typeof(CUDA.CuArray(rand(Float64,2,3)))
                else
                    @test typeof(@rand(2,3,eltype=Float64)) == typeof(parentmodule($package).rand(Float64,2,3))
                end
            end;
            @reset_parallel_kernel()
        end;
        # @testset "3. Exceptions (no default numbertype)" begin
        #     @require !@is_initialized()
        #     @init_parallel_kernel(package = $package)
        #     @require @is_initialized()
        #     @require @get_numbertype() == $NUMBERTYPE_NONE
        #     @testset "no numbertype" begin
        #         @test_throws ArgumentError @zeros(2,3)
        #         @test_throws ArgumentError @ones(2,3)
        #         @test_throws ArgumentError @rand(2,3)
        #     end;
        #     @reset_parallel_kernel()
        # end;
    end;
)) end == nothing || true;
