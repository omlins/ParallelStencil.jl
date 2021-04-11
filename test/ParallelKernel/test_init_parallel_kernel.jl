using Test
import ParallelStencil
using ParallelStencil.ParallelKernel
import ParallelStencil.ParallelKernel: @reset_parallel_kernel, @is_initialized, @get_package, @get_numbertype, SUPPORTED_PACKAGES, PKG_CUDA
import ParallelStencil.ParallelKernel: @require, @symbols, longnameof
import ParallelStencil.ParallelKernel: checkargs_init, check_already_initialized, set_initialized, is_initialized, check_initialized
using ParallelStencil.ParallelKernel.Exceptions
TEST_PACKAGES = SUPPORTED_PACKAGES
@static if PKG_CUDA in TEST_PACKAGES
    import CUDA
    if !CUDA.functional() TEST_PACKAGES = filter!(x->x≠PKG_CUDA, TEST_PACKAGES) end
end

@static for package in TEST_PACKAGES  eval(:(
    @testset "$(basename(@__FILE__)) (package: $(nameof($package)))" begin
        @testset "1. initialization of ParallelKernel" begin
            @require !@is_initialized()
            @init_parallel_kernel($package, ComplexF16)
            @testset "initialized" begin
                @test @is_initialized()
                @test @get_package() == $package
                @test @get_numbertype() == ComplexF16
            end;
            @testset "Data" begin
                @test @isdefined(Data)
                @test length(@symbols($(@__MODULE__), Data)) > 1
                @test Symbol("Number") in @symbols($(@__MODULE__), Data)
                @test Symbol("Array") in @symbols($(@__MODULE__), Data)
                @test Symbol("DeviceArray") in @symbols($(@__MODULE__), Data)
            end;
            @reset_parallel_kernel()
        end;
        @testset "2. Exceptions" begin
            @testset "already initialized" begin
                set_initialized(true)
                @require is_initialized()
                @test_throws IncoherentCallError check_already_initialized()
                set_initialized(false)
            end;
            @testset "arguments" begin
                MyType = Float32
                @test_throws ArgumentError checkargs_init(5, Float64)
                @test_throws ArgumentError checkargs_init($package, MyType)
            end;
            @testset "check_initialized" begin
                @require !is_initialized()
                @test_throws NotInitializedError check_initialized()
            end;
            @reset_parallel_kernel()
        end;
    end;
)) end == nothing || true;
