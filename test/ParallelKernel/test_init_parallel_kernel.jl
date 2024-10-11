using Test
import ParallelStencil
using ParallelStencil.ParallelKernel
import ParallelStencil.ParallelKernel: @reset_parallel_kernel, @is_initialized, @get_package, @get_numbertype, @get_inbounds, NUMBERTYPE_NONE, SUPPORTED_PACKAGES, PKG_CUDA, PKG_AMDGPU
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
Base.retry_load_extensions() # Potentially needed to load the extensions after the packages have been filtered.

@static for package in TEST_PACKAGES  eval(:(
    @testset "$(basename(@__FILE__)) (package: $(nameof($package)))" begin
        @testset "1. initialization of ParallelKernel" begin
            @require !@is_initialized()
            @init_parallel_kernel($package, ComplexF16)
            @testset "initialized" begin
                @test @is_initialized()
                @test @get_package() == $package
                @test @get_numbertype() == ComplexF16
                @test @get_inbounds() == false
            end;
            @testset "Data" begin
                @test @isdefined(Data)
                @test length(@symbols($(@__MODULE__), Data)) > 1
                @test Symbol("Index") in @symbols($(@__MODULE__), Data)
                @test Symbol("Number") in @symbols($(@__MODULE__), Data)
                @test Symbol("Array") in @symbols($(@__MODULE__), Data)
                @test Symbol("Cell") in @symbols($(@__MODULE__), Data)
                @test Symbol("CellArray") in @symbols($(@__MODULE__), Data)
                @testset "Data.Device" begin
                    @test @isdefined(Data.Device)
                    @test length(@symbols($(@__MODULE__), Data.Device)) > 1
                    @test Symbol("Index") in @symbols($(@__MODULE__), Data.Device)
                    @test Symbol("Array") in @symbols($(@__MODULE__), Data.Device)
                    @test Symbol("Cell") in @symbols($(@__MODULE__), Data.Device)
                    @test Symbol("CellArray") in @symbols($(@__MODULE__), Data.Device)
                end;
                @testset "Data.Fields" begin
                    @test @isdefined(Data.Fields)
                    @test length(@symbols($(@__MODULE__), Data.Fields)) > 1
                    @test Symbol("Field") in @symbols($(@__MODULE__), Data.Fields)
                    @test Symbol("VectorField") in @symbols($(@__MODULE__), Data.Fields)
                    @test Symbol("BVectorField") in @symbols($(@__MODULE__), Data.Fields)
                    @test Symbol("TensorField") in @symbols($(@__MODULE__), Data.Fields)
                end;
                @testset "Data.Fields.Device" begin
                    @test @isdefined(Data.Fields.Device)
                    @test length(@symbols($(@__MODULE__), Data.Fields.Device)) > 1
                    @test Symbol("Field") in @symbols($(@__MODULE__), Data.Fields.Device)
                    @test Symbol("VectorField") in @symbols($(@__MODULE__), Data.Fields.Device)
                    @test Symbol("BVectorField") in @symbols($(@__MODULE__), Data.Fields.Device)
                    @test Symbol("TensorField") in @symbols($(@__MODULE__), Data.Fields.Device)
                end;
            end;
            @testset "TData" begin
                @test @isdefined(TData)
                @test length(@symbols($(@__MODULE__), TData)) > 1
                @test Symbol("Index") in @symbols($(@__MODULE__), TData)
                @test Symbol("Number") in @symbols($(@__MODULE__), TData)
                @test Symbol("Array") in @symbols($(@__MODULE__), TData)
                @test Symbol("Cell") in @symbols($(@__MODULE__), TData)
                @test Symbol("CellArray") in @symbols($(@__MODULE__), TData)
                @testset "TData.Device" begin
                    @test @isdefined(TData.Device)
                    @test length(@symbols($(@__MODULE__), TData.Device)) > 1
                    @test Symbol("Index") in @symbols($(@__MODULE__), TData.Device)
                    @test Symbol("Array") in @symbols($(@__MODULE__), TData.Device)
                    @test Symbol("Cell") in @symbols($(@__MODULE__), TData.Device)
                    @test Symbol("CellArray") in @symbols($(@__MODULE__), TData.Device)
                end;
                @testset "TData.Fields" begin
                    @test @isdefined(TData.Fields)
                    @test length(@symbols($(@__MODULE__), TData.Fields)) > 1
                    @test Symbol("Field") in @symbols($(@__MODULE__), TData.Fields)
                    @test Symbol("VectorField") in @symbols($(@__MODULE__), TData.Fields)
                    @test Symbol("BVectorField") in @symbols($(@__MODULE__), TData.Fields)
                    @test Symbol("TensorField") in @symbols($(@__MODULE__), TData.Fields)
                end;
                @testset "TData.Fields.Device" begin
                    @test @isdefined(TData.Fields.Device)
                    @test length(@symbols($(@__MODULE__), TData.Fields.Device)) > 1
                    @test Symbol("Field") in @symbols($(@__MODULE__), TData.Fields.Device)
                    @test Symbol("VectorField") in @symbols($(@__MODULE__), TData.Fields.Device)
                    @test Symbol("BVectorField") in @symbols($(@__MODULE__), TData.Fields.Device)
                    @test Symbol("TensorField") in @symbols($(@__MODULE__), TData.Fields.Device)
                end;
            end;
            @reset_parallel_kernel()
        end;
        @testset "2. initialization of ParallelKernel without numbertype, with inbounds" begin
            @require !@is_initialized()
            @init_parallel_kernel(package = $package, inbounds = true)
            @testset "initialized" begin
                @test @is_initialized()
                @test @get_package() == $package
                @test @get_numbertype() == NUMBERTYPE_NONE
                @test @get_inbounds() == true
            end;
            @testset "Data" begin
                @test @isdefined(Data)
                @test length(@symbols($(@__MODULE__), Data)) > 1
                @test Symbol("Index") in @symbols($(@__MODULE__), Data)
                @test !(Symbol("Number") in @symbols($(@__MODULE__), Data))
                @test Symbol("Array") in @symbols($(@__MODULE__), Data)
                @test Symbol("Cell") in @symbols($(@__MODULE__), Data)
                @test Symbol("CellArray") in @symbols($(@__MODULE__), Data)
                @testset "Data.Device" begin
                    @test @isdefined(Data.Device)
                    @test length(@symbols($(@__MODULE__), Data.Device)) > 1
                    @test Symbol("Index") in @symbols($(@__MODULE__), Data.Device)
                    @test Symbol("Array") in @symbols($(@__MODULE__), Data.Device)
                    @test Symbol("Cell") in @symbols($(@__MODULE__), Data.Device)
                    @test Symbol("CellArray") in @symbols($(@__MODULE__), Data.Device)
                end;
                @testset "Data.Fields" begin
                    @test @isdefined(Data.Fields)
                    @test length(@symbols($(@__MODULE__), Data.Fields)) > 1
                    @test Symbol("Field") in @symbols($(@__MODULE__), Data.Fields)
                    @test Symbol("VectorField") in @symbols($(@__MODULE__), Data.Fields)
                    @test Symbol("BVectorField") in @symbols($(@__MODULE__), Data.Fields)
                    @test Symbol("TensorField") in @symbols($(@__MODULE__), Data.Fields)
                end;
                @testset "Data.Fields.Device" begin
                    @test @isdefined(Data.Fields.Device)
                    @test length(@symbols($(@__MODULE__), Data.Fields.Device)) > 1
                    @test Symbol("Field") in @symbols($(@__MODULE__), Data.Fields.Device)
                    @test Symbol("VectorField") in @symbols($(@__MODULE__), Data.Fields.Device)
                    @test Symbol("BVectorField") in @symbols($(@__MODULE__), Data.Fields.Device)
                    @test Symbol("TensorField") in @symbols($(@__MODULE__), Data.Fields.Device)
                end;
            end;
            @reset_parallel_kernel()
        end;
        @testset "3. Exceptions" begin
            @testset "already initialized" begin
                set_initialized(@__MODULE__, true)
                @require is_initialized(@__MODULE__)
                @test_throws IncoherentCallError check_already_initialized(@__MODULE__)
                set_initialized(@__MODULE__, false)
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
                @require !is_initialized(@__MODULE__)
                @test_throws NotInitializedError check_initialized(@__MODULE__)
            end;
            @reset_parallel_kernel()
        end;
    end;
)) end == nothing || true;
