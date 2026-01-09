using Test
using ParallelStencil
import ParallelStencil: @reset_parallel_stencil, @is_initialized, @get_package, @get_numbertype, @get_ndims, SUPPORTED_PACKAGES, PKG_CUDA, PKG_AMDGPU, PKG_METAL, PKG_POLYESTER, PKG_NONE, NUMBERTYPE_NONE, NDIMS_NONE, select_hardware, current_hardware
import ParallelStencil: @require, @symbols

const PKG_KERNELABSTRACTIONS = hasproperty(ParallelStencil, :PKG_KERNELABSTRACTIONS) ? ParallelStencil.PKG_KERNELABSTRACTIONS : Symbol(:KernelAbstractions)

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

kernelabstractions_runtime_sequence() = Symbol[:cpu]

@static if PKG_KERNELABSTRACTIONS in TEST_PACKAGES
    function kernelabstractions_runtime_sequence()
        sequence = Symbol[:cpu]
        gpu_targets = Symbol[]
        if isdefined(@__MODULE__, :CUDA) && CUDA.functional()
            push!(gpu_targets, :gpu_cuda)
        end
        if isdefined(@__MODULE__, :AMDGPU) && AMDGPU.functional()
            push!(gpu_targets, :gpu_amd)
        end
        if isdefined(@__MODULE__, :Metal) && Sys.isapple() && Metal.functional()
            push!(gpu_targets, :gpu_metal)
        end
        if isempty(gpu_targets)
            push!(sequence, :cpu)
        else
            push!(sequence, gpu_targets[1])
        end
        return sequence
    end
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
                if $package == $PKG_KERNELABSTRACTIONS
                    sequence = kernelabstractions_runtime_sequence()
                    @test !isempty(sequence)
                    @test current_hardware() == sequence[1]
                    for target in sequence[2:end]
                        select_hardware(target)
                        @test current_hardware() == target
                    end
                end
                @reset_parallel_stencil()
                @test length(@symbols($(@__MODULE__), Data)) == 1
                @test !@is_initialized()
                @test @get_package() == $PKG_NONE
                @test @get_numbertype() == $NUMBERTYPE_NONE
                @test @get_ndims() == $NDIMS_NONE
                if $package == $PKG_KERNELABSTRACTIONS
                    @test current_hardware() == :cpu
                end
            end;
        end;
    end;
)) end == nothing || true;
