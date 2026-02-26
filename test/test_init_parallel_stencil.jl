using Test
using ParallelStencil
import ParallelStencil: @reset_parallel_stencil, @is_initialized, @get_package, @get_numbertype, @get_ndims, @get_inbounds, @get_padding, @get_memopt, @get_nonconst_metadata, SUPPORTED_PACKAGES, PKG_CUDA, PKG_AMDGPU, PKG_METAL, PKG_POLYESTER, PKG_KERNELABSTRACTIONS, PKG_NONE, NUMBERTYPE_NONE, NDIMS_NONE, @select_hardware, @current_hardware
import ParallelStencil: @require, @symbols
import ParallelStencil: extract_posargs_init, extract_kwargs_init, check_already_initialized, set_initialized, is_initialized, check_initialized, set_package, set_numbertype, set_ndims, set_inbounds, set_padding, set_memopt, set_nonconst_metadata
using ParallelStencil.Exceptions
import ParallelStencil.ParallelKernel: handle, @isdefined_at_pt
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
        @testset "1. initialization of ParallelStencil" begin
            @require !@is_initialized()
            @init_parallel_stencil($package, ComplexF32, 3)
            @testset "initialized" begin
                @test @is_initialized()
                @test @get_package() == $package
                @test @get_numbertype() == ComplexF32
                @test @get_ndims() == 3
                @test @get_memopt() == false
                @test @get_nonconst_metadata() == false
                @test @get_inbounds() == false
                @test @get_padding() == false
            end;
            @testset "default hardware" begin
                parse_hw = ParallelStencil.ParallelKernel.@get_hardware()
                @static if $package == $PKG_KERNELABSTRACTIONS
                    @test parse_hw == :cpu
                elseif $package == $PKG_CUDA
                    @test parse_hw == :gpu_cuda
                elseif $package == $PKG_AMDGPU
                    @test parse_hw == :gpu_amd
                elseif $package == $PKG_METAL
                    @test parse_hw == :gpu_metal
                else
                    @test parse_hw == :cpu
                end
            end;
            @static if $package == $PKG_KERNELABSTRACTIONS
                @testset "KernelAbstractions exposes no Data modules" begin
                    @test !@isdefined_at_pt(Data) || (@isdefined_at_pt(Data) && !@isdefined_at_pt(Data.Device))
                    @test !@isdefined_at_pt(TData) || (@isdefined_at_pt(TData) && !@isdefined_at_pt(TData.Device))
                end;
                @select_hardware(:cpu)
                @test @current_hardware() == :cpu
            else
                @testset "Data" begin
                    @test @isdefined_at_pt(Data)
                    @test length(@symbols($(@__MODULE__), Data)) > 1
                    @testset "Data.Device" begin
                        @test @isdefined_at_pt(Data.Device)
                        @test length(@symbols($(@__MODULE__), Data.Device)) > 1
                    end;
                    @testset "Data.Fields" begin
                        @test @isdefined_at_pt(Data.Fields)
                        @test length(@symbols($(@__MODULE__), Data.Fields)) > 1
                    end;
                    @testset "Data.Fields.Device" begin
                        @test @isdefined_at_pt(Data.Fields.Device)
                        @test length(@symbols($(@__MODULE__), Data.Fields.Device)) > 1
                    end;
                end;
                @testset "TData" begin
                    @test @isdefined_at_pt(TData)
                    @test length(@symbols($(@__MODULE__), TData)) > 1
                    @testset "TData.Device" begin
                        @test @isdefined_at_pt(TData.Device)
                        @test length(@symbols($(@__MODULE__), TData.Device)) > 1
                    end;
                    @testset "TData.Fields" begin
                        @test @isdefined_at_pt(TData.Fields)
                        @test length(@symbols($(@__MODULE__), TData.Fields)) > 1
                    end;
                    @testset "TData.Fields.Device" begin
                        @test @isdefined_at_pt(TData.Fields.Device)
                        @test length(@symbols($(@__MODULE__), TData.Fields.Device)) > 1
                    end;
                end;
            end
            @reset_parallel_stencil()
        end;
        @testset "2. initialization of ParallelStencil without numbertype and ndims, with memopt, inbounds and padding (and nonconst_metadata)" begin
            @require !@is_initialized()
            @init_parallel_stencil(package = $package, inbounds = true, padding = false, memopt = true, nonconst_metadata = true)
            @testset "initialized" begin
                @test @is_initialized()
                @test @get_package() == $package
                @test @get_numbertype() == NUMBERTYPE_NONE
                @test @get_ndims() == NDIMS_NONE
                @test @get_memopt() == true
                @test @get_nonconst_metadata() == true
                @test @get_inbounds() == true
                @test @get_padding() == false   #TODO: this needs to be restored to true when Polyester supports padding.
            end;
            @testset "default hardware" begin
                parse_hw = ParallelStencil.ParallelKernel.@get_hardware()
                @static if $package == $PKG_KERNELABSTRACTIONS
                    @test parse_hw == :cpu
                elseif $package == $PKG_CUDA
                    @test parse_hw == :gpu_cuda
                elseif $package == $PKG_AMDGPU
                    @test parse_hw == :gpu_amd
                elseif $package == $PKG_METAL
                    @test parse_hw == :gpu_metal
                else
                    @test parse_hw == :cpu
                end
            end;
            @static if $package == $PKG_KERNELABSTRACTIONS
                @testset "KernelAbstractions exposes no Data modules" begin
                    @test !@isdefined_at_pt(Data) || (@isdefined_at_pt(Data) && !@isdefined_at_pt(Data.Device))
                    @test !@isdefined_at_pt(TData) || (@isdefined_at_pt(TData) && !@isdefined_at_pt(TData.Device))
                end;
                @select_hardware(:cpu)
                @test @current_hardware() == :cpu
            else
                @testset "Data" begin
                    @test @isdefined_at_pt(Data)
                    @test length(@symbols($(@__MODULE__), Data)) > 1
                    @testset "Data.Device" begin
                        @test @isdefined_at_pt(Data.Device)
                        @test length(@symbols($(@__MODULE__), Data.Device)) > 1
                    end;
                    @testset "Data.Fields" begin
                        @test @isdefined_at_pt(Data.Fields)
                        @test length(@symbols($(@__MODULE__), Data.Fields)) > 1
                    end;
                    @testset "Data.Fields.Device" begin
                        @test @isdefined_at_pt(Data.Fields.Device)
                        @test length(@symbols($(@__MODULE__), Data.Fields.Device)) > 1
                    end;
                end;
            end
            @reset_parallel_stencil()
        end;
        @testset "3. Exceptions" begin
            @init_parallel_stencil(package=$package) # NOTE: Initialization is potentially later required to create the metadata module
            @testset "already initialized" begin
                set_initialized(@__MODULE__, true)
                set_package(@__MODULE__, :CUDA)
                set_numbertype(@__MODULE__, Float64)
                set_ndims(@__MODULE__, 3)
                set_memopt(@__MODULE__, false)
                set_inbounds(@__MODULE__, false)
                set_padding(@__MODULE__, false)
                set_nonconst_metadata(@__MODULE__, false)
                @require is_initialized(@__MODULE__)
                @test_throws IncoherentCallError check_already_initialized(@__MODULE__, :Threads, Float64, 3, false, false, false, false)
                @test_throws IncoherentCallError check_already_initialized(@__MODULE__, :CUDA, Float32, 3, false, false, false, false)
                @test_throws IncoherentCallError check_already_initialized(@__MODULE__, :CUDA, Float64, 2, false, false, false, false)
                @test_throws IncoherentCallError check_already_initialized(@__MODULE__, :CUDA, Float64, 3, true, false, false, false)
                @test_throws IncoherentCallError check_already_initialized(@__MODULE__, :CUDA, Float64, 3, false, true, false, false)
                @test_throws IncoherentCallError check_already_initialized(@__MODULE__, :CUDA, Float64, 3, false, false, true, false)
                @test_throws IncoherentCallError check_already_initialized(@__MODULE__, :AMDGPU, Float16, 1, true, false, true, false)
                set_initialized(@__MODULE__, false)
                set_package(@__MODULE__, PKG_NONE)
                set_numbertype(@__MODULE__, NUMBERTYPE_NONE)
                set_ndims(@__MODULE__, NDIMS_NONE)
            end;
            @testset "arguments" begin
                @test_throws ArgumentError extract_posargs_init($(@__MODULE__), 99, :Float64, 3)
                @test_throws ArgumentError extract_posargs_init($(@__MODULE__), nameof($package), :Char, 3)
                @test_throws ArgumentError extract_posargs_init($(@__MODULE__), nameof($package), :Float64, 77)
                @test_throws ArgumentEvaluationError extract_posargs_init($(@__MODULE__), nameof($package), :MyType, 3)
                @test_throws ArgumentEvaluationError extract_posargs_init($(@__MODULE__), nameof($package), :Float64, :myndims)
                @test_throws ArgumentError extract_kwargs_init($(@__MODULE__), Dict(:package => 99, :numbertype => :Float64, :ndims => 3))
                @test_throws ArgumentError extract_kwargs_init($(@__MODULE__), Dict(:package => nameof($package), :numbertype => :Char, :ndims => 3))
                @test_throws ArgumentError extract_kwargs_init($(@__MODULE__), Dict(:package => nameof($package), :numbertype => :Float64, :ndims => 77))
                @test_throws ArgumentEvaluationError extract_kwargs_init($(@__MODULE__), Dict(:package => nameof($package), :numbertype => :MyType, :ndims => 3))
                @test_throws ArgumentEvaluationError extract_kwargs_init($(@__MODULE__), Dict(:package => nameof($package), :numbertype => :MyType, :ndims => :myndims))
                @test_throws ArgumentError extract_kwargs_init($(@__MODULE__), Dict(:ndims => 77))
                @test_throws ArgumentEvaluationError extract_kwargs_init($(@__MODULE__), Dict(:ndims => :myndims))
            end;
            @testset "check_initialized" begin
                @require !is_initialized(@__MODULE__)
                @test_throws NotInitializedError check_initialized(@__MODULE__)
            end;
            @reset_parallel_stencil()
         end;
    end;
)) end == nothing || true;
