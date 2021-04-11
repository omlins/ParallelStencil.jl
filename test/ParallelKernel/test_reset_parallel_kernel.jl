using Test
import ParallelStencil
using ParallelStencil.ParallelKernel
import ParallelStencil.ParallelKernel: @reset_parallel_kernel, @is_initialized, @get_package, @get_numbertype, SUPPORTED_PACKAGES, PKG_CUDA, PKG_NONE, NUMBERTYPE_NONE
import ParallelStencil.ParallelKernel: @require, @symbols, longnameof
TEST_PACKAGES = SUPPORTED_PACKAGES
@static if PKG_CUDA in TEST_PACKAGES
    import CUDA
    if !CUDA.functional() TEST_PACKAGES = filter!(x->x≠PKG_CUDA, TEST_PACKAGES) end
end

@static for package in TEST_PACKAGES  eval(:(
    @testset "$(basename(@__FILE__)) (package: $(nameof($package)))" begin
        @testset "1. Reset of ParallelKernel" begin
            @testset "Reset if not initialized" begin
                @require !@is_initialized()
                @reset_parallel_kernel()
                @test !@is_initialized()
                @test @get_package() == $PKG_NONE
                @test @get_numbertype() == $NUMBERTYPE_NONE
            end;
            @testset "Reset if initialized" begin
                @require !@is_initialized()
                @init_parallel_kernel($package, Float64)
                @require @is_initialized() && @get_package() == $package
                @reset_parallel_kernel()
                @test length(@symbols($(@__MODULE__), Data)) == 1
                @test !@is_initialized()
                @test @get_package() == $PKG_NONE
                @test @get_numbertype() == $NUMBERTYPE_NONE
            end;
        end;
    end;
)) end == nothing || true;
