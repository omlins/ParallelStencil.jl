using Test
import ParallelStencil
using ParallelStencil.ParallelKernel
import ParallelStencil.ParallelKernel: @reset_parallel_kernel, @is_initialized, @get_package, @get_numbertype, @get_inbounds, @get_padding, NUMBERTYPE_NONE, SUPPORTED_PACKAGES, PKG_CUDA, PKG_AMDGPU, PKG_METAL, PKG_POLYESTER, PKG_THREADS, SCALARTYPES, ARRAYTYPES, FIELDTYPES
import ParallelStencil.ParallelKernel: @require, @symbols
import ParallelStencil.ParallelKernel: select_hardware, current_hardware
import ParallelStencil.ParallelKernel: extract_posargs_init, extract_kwargs_init, check_already_initialized, set_initialized, is_initialized, check_initialized
using ParallelStencil.ParallelKernel.Exceptions
const PKG_KERNELABSTRACTIONS = hasproperty(ParallelStencil.ParallelKernel, :PKG_KERNELABSTRACTIONS) ? ParallelStencil.ParallelKernel.PKG_KERNELABSTRACTIONS : Symbol(:KernelAbstractions)

TEST_PACKAGES = collect(SUPPORTED_PACKAGES)
if PKG_KERNELABSTRACTIONS ∉ TEST_PACKAGES
    push!(TEST_PACKAGES, PKG_KERNELABSTRACTIONS)
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
@static if PKG_KERNELABSTRACTIONS in TEST_PACKAGES
    if Base.find_package("KernelAbstractions") === nothing
        TEST_PACKAGES = filter!(x->x≠PKG_KERNELABSTRACTIONS, TEST_PACKAGES)
    else
        import KernelAbstractions
    end
end
Base.retry_load_extensions() # Potentially needed to load the extensions after the packages have been filtered.

function default_hardware_symbol(package::Symbol)
    if package == PKG_KERNELABSTRACTIONS
        return :cpu
    elseif package == PKG_THREADS || package == PKG_POLYESTER
        return :cpu
    elseif package == PKG_CUDA
        return :gpu_cuda
    elseif package == PKG_AMDGPU
        return :gpu_amd
    elseif package == PKG_METAL
        return :gpu_metal
    else
        return :cpu
    end
end

function invalid_hardware_symbol(package::Symbol, default_symbol::Symbol)
    if package == PKG_THREADS || package == PKG_POLYESTER
        return :gpu_cuda
    elseif package == PKG_CUDA
        return :cpu
    elseif package == PKG_AMDGPU
        return :cpu
    elseif package == PKG_METAL
        return :cpu
    else
        return Symbol(default_symbol, :_unsupported)
    end
end

is_multi_architecture(package::Symbol) = package == PKG_KERNELABSTRACTIONS


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
                @test @get_padding() == false
            end;
            runtime_default = default_hardware_symbol($package)
            @testset "Runtime hardware defaults" begin
                @test current_hardware() == runtime_default
                if is_multi_architecture($package)
                    @test_throws ArgumentError select_hardware(:unsupported_hardware_symbol)
                    @test current_hardware() == runtime_default
                else
                    invalid_symbol = invalid_hardware_symbol($package, runtime_default)
                    @test_throws ArgumentError select_hardware(invalid_symbol)
                    @test current_hardware() == runtime_default
                end
            end;
            if $package == PKG_KERNELABSTRACTIONS
                @testset "Data" begin
                    @test !@isdefined(Data)
                end;
                @testset "TData" begin
                    @test !@isdefined(TData)
                end;
            else
                @testset "Data" begin
                    @test @isdefined(Data)
                    mods = (:Data, :Device, :Fields)
                    syms = @symbols($(@__MODULE__), Data)
                    @test length(syms) > 1
                    @test length(syms) >= length(mods) + length(SCALARTYPES) + length(ARRAYTYPES) # +1|2 for metadata symbols
                    @test all(T ∈ syms for T in mods)
                    @test all(T ∈ syms for T in SCALARTYPES)
                    @test all(T ∈ syms for T in ARRAYTYPES)
                    @testset "Data.Device" begin
                        syms = @symbols($(@__MODULE__), Data.Device)
                        @test length(syms) > 0
                        @test all(T ∈ syms for T in ARRAYTYPES)
                    end;
                    @testset "Data.Fields" begin
                        mods = (:Fields, :Device)
                        syms = @symbols($(@__MODULE__), Data.Fields)
                        @test length(syms) > 0
                        @test all(T ∈ syms for T in mods)
                        @test all(T ∈ syms for T in FIELDTYPES)
                    end;
                    @testset "Data.Fields.Device" begin
                        syms = @symbols($(@__MODULE__), Data.Fields.Device)
                        @test length(syms) > 0
                        @test all(T ∈ syms for T in FIELDTYPES)
                    end;
                end;
                @testset "TData" begin # NOTE: no scalar types
                    @test @isdefined(TData)
                    mods = (:TData, :Device, :Fields)
                    syms = @symbols($(@__MODULE__), TData)
                    @test length(syms) > 1
                    @test all(T ∈ syms for T in mods)
                    @test all(T ∈ syms for T in ARRAYTYPES)
                    @testset "TData.Device" begin
                        syms = @symbols($(@__MODULE__), TData.Device)
                        @test length(syms) > 0
                        @test all(T ∈ syms for T in ARRAYTYPES)
                    end;
                    @testset "TData.Fields" begin
                        mods = (:Fields, :Device)
                        syms = @symbols($(@__MODULE__), TData.Fields)
                        @test length(syms) > 0
                        @test all(T ∈ syms for T in mods)
                        @test all(T ∈ syms for T in FIELDTYPES)
                    end;
                    @testset "TData.Fields.Device" begin
                        syms = @symbols($(@__MODULE__), TData.Fields.Device)
                        @test length(syms) > 0
                        @test all(T ∈ syms for T in FIELDTYPES)
                    end;
                end;
            end;
            @reset_parallel_kernel()
        end;
        @testset "2. initialization of ParallelKernel without numbertype, with inbounds and padding" begin
            @require !@is_initialized()
            @init_parallel_kernel(package = $package, inbounds = true, padding = true)
            @testset "initialized" begin
                @test @is_initialized()
                @test @get_package() == $package
                @test @get_numbertype() == NUMBERTYPE_NONE
                @test @get_inbounds() == true
                @test @get_padding() == true
                if $package == PKG_KERNELABSTRACTIONS
                    @test current_hardware() == :cpu
                end
            end;
            if $package == PKG_KERNELABSTRACTIONS
                @testset "Data" begin # NOTE: not generated for KernelAbstractions
                    @test !@isdefined(Data)
                end;
            else
                @testset "Data" begin # NOTE: no scalar types
                    @test @isdefined(Data)
                    mods = (:Data, :Device, :Fields)
                    syms = @symbols($(@__MODULE__), Data)
                    @test length(syms) > 1
                    @test all(T ∈ syms for T in mods)
                    @test !(Symbol("Number") in syms)
                    @test all(T ∈ syms for T in ARRAYTYPES)
                    @testset "Data.Device" begin
                        syms = @symbols($(@__MODULE__), Data.Device)
                        @test length(syms) > 0
                        @test all(T ∈ syms for T in ARRAYTYPES)
                    end;
                    @testset "Data.Fields" begin
                        mods = (:Fields, :Device)
                        syms = @symbols($(@__MODULE__), Data.Fields)
                        @test length(syms) > 0
                        @test all(T ∈ syms for T in mods)
                        @test all(T ∈ syms for T in FIELDTYPES)
                    end;
                    @testset "Data.Fields.Device" begin
                        syms = @symbols($(@__MODULE__), Data.Fields.Device)
                        @test length(syms) > 0
                        @test all(T ∈ syms for T in FIELDTYPES)
                    end;
                end;
            end
            @reset_parallel_kernel()
        end;
        @testset "3. Exceptions" begin
            @init_parallel_kernel(package=$package) # NOTE: Initialization is potentially later required to create the metadata module
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
