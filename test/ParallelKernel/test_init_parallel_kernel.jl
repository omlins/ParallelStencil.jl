using Test
import ParallelStencil
using ParallelStencil.ParallelKernel
import ParallelStencil.ParallelKernel: @reset_parallel_kernel, @is_initialized, @get_package, @get_numbertype, NUMBERTYPE_NONE, SUPPORTED_PACKAGES, PKG_CUDA, PKG_AMDGPU
import ParallelStencil.ParallelKernel: @require, @symbols
import ParallelStencil.ParallelKernel: extract_posargs_init, extract_kwargs_init, check_already_initialized, set_initialized, is_initialized, check_initialized
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
                @test Symbol("Cell") in @symbols($(@__MODULE__), Data)
                @test Symbol("CellArray") in @symbols($(@__MODULE__), Data)
                @test Symbol("DeviceArray") in @symbols($(@__MODULE__), Data)
                @test Symbol("DeviceCell") in @symbols($(@__MODULE__), Data)
                @test Symbol("DeviceCellArray") in @symbols($(@__MODULE__), Data)
                @test Symbol("TArray") in @symbols($(@__MODULE__), Data)
                @test Symbol("TCell") in @symbols($(@__MODULE__), Data)
                @test Symbol("TCellArray") in @symbols($(@__MODULE__), Data)
                @test Symbol("DeviceTArray") in @symbols($(@__MODULE__), Data)
                @test Symbol("DeviceTCell") in @symbols($(@__MODULE__), Data)
                @test Symbol("DeviceTCellArray") in @symbols($(@__MODULE__), Data)
            end;
            @reset_parallel_kernel()
        end;
        @testset "2. initialization of ParallelKernel without numbertype" begin
            @require !@is_initialized()
            @init_parallel_kernel(package = $package)
            @testset "initialized" begin
                @test @is_initialized()
                @test @get_package() == $package
                @test @get_numbertype() == NUMBERTYPE_NONE
            end;
            @testset "Data" begin
                @test @isdefined(Data)
                @test length(@symbols($(@__MODULE__), Data)) > 1
                @test !(Symbol("Number") in @symbols($(@__MODULE__), Data))
                @test Symbol("Array") in @symbols($(@__MODULE__), Data)
                @test Symbol("Cell") in @symbols($(@__MODULE__), Data)
                @test Symbol("CellArray") in @symbols($(@__MODULE__), Data)
                @test Symbol("DeviceArray") in @symbols($(@__MODULE__), Data)
                @test Symbol("DeviceCell") in @symbols($(@__MODULE__), Data)
                @test Symbol("DeviceCellArray") in @symbols($(@__MODULE__), Data)
            end;
            @reset_parallel_kernel()
        end;
        @testset "3. Exceptions" begin
            @testset "already initialized" begin
                set_initialized(true)
                @require is_initialized()
                @test_throws IncoherentCallError check_already_initialized()
                set_initialized(false)
            end;
            @testset "arguments" begin
                @test_throws ArgumentError extract_posargs_init($(@__MODULE__), 99, :Float64)
                @test_throws ArgumentError extract_posargs_init($(@__MODULE__), nameof($package), :Char)
                @test_throws ArgumentEvaluationError extract_posargs_init($(@__MODULE__), nameof($package), :MyType)
                @test_throws ArgumentError extract_kwargs_init($(@__MODULE__), Dict(:package => 99, :numbertype => :Float64))
                @test_throws ArgumentError extract_kwargs_init($(@__MODULE__), Dict(:package => nameof($package), :numbertype => :Char))
                @test_throws ArgumentEvaluationError extract_kwargs_init($(@__MODULE__), Dict(:package => nameof($package), :numbertype => :MyType))
                @test_throws ArgumentError extract_kwargs_init($(@__MODULE__), Dict(:numbertype => :Char))
                @test_throws ArgumentEvaluationError extract_kwargs_init($(@__MODULE__), Dict(:numbertype => :MyType))
            end;
            @testset "check_initialized" begin
                @require !is_initialized()
                @test_throws NotInitializedError check_initialized()
            end;
            @reset_parallel_kernel()
        end;
    end;
)) end == nothing || true;
