using Test
import ParallelStencil
using ParallelStencil.ParallelKernel
import ParallelStencil.ParallelKernel: @reset_parallel_kernel, @is_initialized, @get_package, @get_numbertype, @get_hardware, @get_inbounds, @get_padding, NUMBERTYPE_NONE, SUPPORTED_PACKAGES, PKG_CUDA, PKG_AMDGPU, PKG_METAL, PKG_POLYESTER, PKG_KERNELABSTRACTIONS, SCALARTYPES, ARRAYTYPES, FIELDTYPES, @select_hardware, @current_hardware, handle
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
@static if PKG_METAL in TEST_PACKAGES
    import Metal
    if !Metal.functional() TEST_PACKAGES = filter!(x->x≠PKG_METAL, TEST_PACKAGES) end
end
@static if PKG_KERNELABSTRACTIONS in TEST_PACKAGES
    import KernelAbstractions
    if !KernelAbstractions.functional(KernelAbstractions.CPU()) TEST_PACKAGES = filter!(x->x≠PKG_KERNELABSTRACTIONS, TEST_PACKAGES) end
end
@static if PKG_POLYESTER in TEST_PACKAGES
    import Polyester
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
                @test @get_padding() == false
            end;
            @testset "default hardware" begin
                default_hw = @get_hardware()
                if $package == $PKG_KERNELABSTRACTIONS
                    @test default_hw == :cpu
                elseif $package == $PKG_CUDA
                    @test default_hw == :gpu_cuda
                elseif $package == $PKG_AMDGPU
                    @test default_hw == :gpu_amd
                elseif $package == $PKG_METAL
                    @test default_hw == :gpu_metal
                else
                    @test default_hw == :cpu
                end
            end;
            if $package == $PKG_KERNELABSTRACTIONS
                @testset "KernelAbstractions exposes no Data modules" begin
                    @test !@isdefined(Data)
                    @test !@isdefined(TData)
                    syms = @symbols($(@__MODULE__), $(@__MODULE__))
                    @require length(filter(sym -> sym in (:Data, :TData), syms)) == 0
                end;
                @select_hardware(:cpu)
                @test @current_hardware() == :cpu
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
                        if isdefined(Data, :Device)
                            syms = names(getfield(Data, :Device), all=true, imported=true)
                            @test length(syms) > 0
                            @test all(T ∈ syms for T in ARRAYTYPES)
                        else
                            @test !isdefined(Data, :Device)
                        end
                    end;
                    @testset "Data.Fields" begin
                        if isdefined(Data, :Fields)
                            mods = (:Fields, :Device)
                            syms = names(getfield(Data, :Fields), all=true, imported=true)
                            @test length(syms) > 0
                            @test all(T ∈ syms for T in mods)
                            @test all(T ∈ syms for T in FIELDTYPES)
                        else
                            @test !isdefined(Data, :Fields)
                        end
                    end;
                    @testset "Data.Fields.Device" begin
                        if isdefined(Data, :Fields) && isdefined(getfield(Data, :Fields), :Device)
                            syms = names(getfield(getfield(Data, :Fields), :Device), all=true, imported=true)
                            @test length(syms) > 0
                            @test all(T ∈ syms for T in FIELDTYPES)
                        else
                            @test !(isdefined(Data, :Fields) && isdefined(getfield(Data, :Fields), :Device))
                        end
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
                        if isdefined(TData, :Device)
                            syms = names(getfield(TData, :Device), all=true, imported=true)
                            @test length(syms) > 0
                            @test all(T ∈ syms for T in ARRAYTYPES)
                        else
                            @test !isdefined(TData, :Device)
                        end
                    end;
                    @testset "TData.Fields" begin
                        if isdefined(TData, :Fields)
                            mods = (:Fields, :Device)
                            syms = names(getfield(TData, :Fields), all=true, imported=true)
                            @test length(syms) > 0
                            @test all(T ∈ syms for T in mods)
                            @test all(T ∈ syms for T in FIELDTYPES)
                        else
                            @test !isdefined(TData, :Fields)
                        end
                    end;
                    @testset "TData.Fields.Device" begin
                        if isdefined(TData, :Fields) && isdefined(getfield(TData, :Fields), :Device)
                            syms = names(getfield(getfield(TData, :Fields), :Device), all=true, imported=true)
                            @test length(syms) > 0
                            @test all(T ∈ syms for T in FIELDTYPES)
                        else
                            @test !(isdefined(TData, :Fields) && isdefined(getfield(TData, :Fields), :Device))
                        end
                    end;
                end;
            end
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
            end;
            @testset "default hardware" begin
                default_hw = @get_hardware()
                if $package == $PKG_KERNELABSTRACTIONS
                    @test default_hw == :cpu
                elseif $package == $PKG_CUDA
                    @test default_hw == :gpu_cuda
                elseif $package == $PKG_AMDGPU
                    @test default_hw == :gpu_amd
                elseif $package == $PKG_METAL
                    @test default_hw == :gpu_metal
                else
                    @test default_hw == :cpu
                end
            end;
            if $package == $PKG_KERNELABSTRACTIONS
                @testset "KernelAbstractions exposes no Data modules" begin
                    @test !@isdefined(Data)
                    @test !@isdefined(TData)
                    syms = @symbols($(@__MODULE__), $(@__MODULE__))
                    @require length(filter(sym -> sym in (:Data, :TData), syms)) == 0
                end;
                @select_hardware(:cpu)
                @test @current_hardware() == :cpu
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
                        if isdefined(Data, :Device)
                            syms = names(getfield(Data, :Device), all=true, imported=true)
                            @test length(syms) > 0
                            @test all(T ∈ syms for T in ARRAYTYPES)
                        else
                            @test !isdefined(Data, :Device)
                        end
                    end;
                    @testset "Data.Fields" begin
                        if isdefined(Data, :Fields)
                            mods = (:Fields, :Device)
                            syms = names(getfield(Data, :Fields), all=true, imported=true)
                            @test length(syms) > 0
                            @test all(T ∈ syms for T in mods)
                            @test all(T ∈ syms for T in FIELDTYPES)
                        else
                            @test !isdefined(Data, :Fields)
                        end
                    end;
                    @testset "Data.Fields.Device" begin
                        if isdefined(Data, :Fields) && isdefined(getfield(Data, :Fields), :Device)
                            syms = names(getfield(getfield(Data, :Fields), :Device), all=true, imported=true)
                            @test length(syms) > 0
                            @test all(T ∈ syms for T in FIELDTYPES)
                        else
                            @test !(isdefined(Data, :Fields) && isdefined(getfield(Data, :Fields), :Device))
                        end
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
