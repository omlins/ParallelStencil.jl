using Test
using ParallelStencil
import ParallelStencil: @reset_parallel_stencil, @is_initialized, @get_package, @get_numbertype, @get_ndims, SUPPORTED_PACKAGES, PKG_CUDA, PKG_NONE, NUMBERTYPE_NONE, NDIMS_NONE
import ParallelStencil: @require, @symbols, longnameof
TEST_PACKAGES = SUPPORTED_PACKAGES
@static if PKG_CUDA in TEST_PACKAGES
    import CUDA
    if !CUDA.functional() TEST_PACKAGES = filter!(x->x≠PKG_CUDA, TEST_PACKAGES) end
end

@static for package in TEST_PACKAGES  eval(:(
    @testset "$(basename(@__FILE__)) (package: $(nameof($package)))" begin
        @testset "1. Reset of ParallelStencil" begin
            @testset "Reset if not initialized" begin
                @require !@is_initialized()
                @reset_parallel_stencil()
                @test !@is_initialized()
                @test @get_package() == $PKG_NONE
                @test @get_numbertype() == $NUMBERTYPE_NONE
                @test @get_ndims() == $NDIMS_NONE
            end;
            @testset "Reset if initialized" begin
                @require !@is_initialized()
                @init_parallel_stencil($package, Float64, 3)
                @require @is_initialized() && @get_package() == $package
                @reset_parallel_stencil()
                @test length(@symbols($(@__MODULE__), Data)) == 1
                @test !@is_initialized()
                @test @get_package() == $PKG_NONE
                @test @get_numbertype() == $NUMBERTYPE_NONE
                @test @get_ndims() == $NDIMS_NONE
            end;
        end;
    end;
)) end == nothing || true;
