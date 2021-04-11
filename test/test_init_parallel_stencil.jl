using Test
using ParallelStencil
import ParallelStencil: @reset_parallel_stencil, @is_initialized, @get_package,  @get_numbertype, @get_ndims, SUPPORTED_PACKAGES, PKG_CUDA, PKG_NONE, NUMBERTYPE_NONE, NDIMS_NONE
import ParallelStencil: @require, @symbols, longnameof
import ParallelStencil: checkargs_init, check_already_initialized, set_initialized, is_initialized, check_initialized, set_package, set_numbertype, set_ndims
using ParallelStencil.Exceptions
TEST_PACKAGES = SUPPORTED_PACKAGES
@static if PKG_CUDA in TEST_PACKAGES
    import CUDA
    if !CUDA.functional() TEST_PACKAGES = filter!(x->x≠PKG_CUDA, TEST_PACKAGES) end
end

@static for package in TEST_PACKAGES  eval(:(
    @testset "$(basename(@__FILE__)) (package: $(nameof($package)))" begin
        @testset "1. initialization of ParallelStencil" begin
            @require !@is_initialized()
            @init_parallel_stencil($package, ComplexF32, 3)
            @testset "initialized" begin
                @test @is_initialized()
                @test @get_package() == $package
                @test @get_numbertype() == ComplexF32
                @test @get_ndims() == 3
            end;
            @testset "Data" begin
                @test @isdefined(Data)
                @test length(@symbols($(@__MODULE__), Data)) > 1
                @test Symbol("Number") in @symbols($(@__MODULE__), Data)
                @test Symbol("Array") in @symbols($(@__MODULE__), Data)
                @test Symbol("DeviceArray") in @symbols($(@__MODULE__), Data)
            end;
            @reset_parallel_stencil()
        end;
        @testset "2. Exceptions" begin
            @testset "already initialized" begin
                set_initialized(true)
                set_package(:CUDA)
                set_numbertype(Float64)
                set_ndims(3)
                @require is_initialized()
                @test_throws IncoherentCallError check_already_initialized(:Threads, Float64, 3)
                @test_throws IncoherentCallError check_already_initialized(:CUDA, Float32, 3)
                @test_throws IncoherentCallError check_already_initialized(:CUDA, Float64, 2)
                set_initialized(false)
                set_package(PKG_NONE)
                set_numbertype(NUMBERTYPE_NONE)
                set_ndims(NDIMS_NONE)
            end;
            @testset "arguments" begin
                MyType = Float32
                @test_throws ArgumentError checkargs_init(5, Float64, 3)
                @test_throws ArgumentError checkargs_init($package, MyType, 3)
                @test_throws ArgumentError checkargs_init($package, Float64, Float64)
            end;
            @testset "check_initialized" begin
                @require !is_initialized()
                @test_throws NotInitializedError check_initialized()
            end;
            @reset_parallel_stencil()
         end;
    end;
)) end == nothing || true;
