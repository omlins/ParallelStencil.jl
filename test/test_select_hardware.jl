using Test
using ParallelStencil
import ParallelStencil: @init_parallel_stencil, @reset_parallel_stencil, @is_initialized, @require, SUPPORTED_PACKAGES
using ParallelStencil.ParallelKernel
import ParallelStencil.ParallelKernel: select_hardware as pk_select_hardware, current_hardware as pk_current_hardware, @reset_parallel_kernel, @is_initialized, @require
import ParallelStencil.ParallelKernel.Exceptions: ArgumentError

const PKG_KERNELABSTRACTIONS = hasproperty(ParallelStencil.ParallelKernel, :PKG_KERNELABSTRACTIONS) ? ParallelStencil.ParallelKernel.PKG_KERNELABSTRACTIONS : Symbol(:KernelAbstractions)

TEST_PACKAGES = Symbol[]
if PKG_KERNELABSTRACTIONS in SUPPORTED_PACKAGES
    push!(TEST_PACKAGES, PKG_KERNELABSTRACTIONS)
else
    push!(TEST_PACKAGES, PKG_KERNELABSTRACTIONS)
end

@static if PKG_KERNELABSTRACTIONS in TEST_PACKAGES
    if Base.find_package("KernelAbstractions") === nothing
        TEST_PACKAGES = Symbol[]
    else
        import KernelAbstractions
    end
end

Base.retry_load_extensions()

function kernelabstractions_runtime_sequence()
    sequence = Symbol[:cpu]
    gpu_targets = Symbol[]
    if Base.find_package("CUDA") !== nothing
        push!(gpu_targets, :gpu_cuda)
    end
    if Base.find_package("AMDGPU") !== nothing
        push!(gpu_targets, :gpu_amd)
    end
    if Sys.isapple() && Base.find_package("Metal") !== nothing
        push!(gpu_targets, :gpu_metal)
    end
    if Base.find_package("oneAPI") !== nothing
        push!(gpu_targets, :gpu_oneapi)
    end
    if !isempty(gpu_targets)
        push!(sequence, gpu_targets[1])
        push!(sequence, :cpu)
    end
    return sequence
end

@static for package in TEST_PACKAGES eval(:(
    @testset "$(basename(@__FILE__)) (package: $(nameof($package)))" begin
        @require !@is_initialized()
        @init_parallel_stencil($package, Float32, 2)
        @test ParallelKernel.@is_initialized()

        runtime_sequence = kernelabstractions_runtime_sequence()
        @test !isempty(runtime_sequence)
        @test pk_current_hardware() == runtime_sequence[1]
        @test ParallelStencil.current_hardware() == runtime_sequence[1]

        for target in runtime_sequence[2:end]
            ParallelStencil.select_hardware(target)
            @test ParallelStencil.current_hardware() == target
            @test pk_current_hardware() == target
        end

        ParallelStencil.select_hardware(:cpu)
        @test ParallelStencil.current_hardware() == :cpu
        @test pk_current_hardware() == :cpu

        err_type = nothing
        try
            pk_select_hardware(:unsupported_hardware_symbol)
        catch err
            err_type = typeof(err)
        end
        @test err_type !== nothing
        @test_throws err_type ParallelStencil.select_hardware(:unsupported_hardware_symbol)
        @test ParallelStencil.current_hardware() == :cpu
        @test pk_current_hardware() == :cpu

        @reset_parallel_stencil()
        @reset_parallel_kernel()
        @test !@is_initialized()
        @test !ParallelKernel.@is_initialized()
    end;
)) end == nothing || true
