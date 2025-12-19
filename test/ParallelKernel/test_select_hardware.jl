using Test
import ParallelStencil
using ParallelStencil.ParallelKernel
import ParallelStencil.ParallelKernel: @init_parallel_kernel, @reset_parallel_kernel, @is_initialized, @require, SUPPORTED_PACKAGES, PKG_CUDA, PKG_AMDGPU, PKG_METAL, PKG_THREADS, PKG_POLYESTER, select_hardware, current_hardware
import ParallelStencil.ParallelKernel: handle
import ParallelStencil.ParallelKernel.Exceptions: ArgumentError

const PKG_KERNELABSTRACTIONS = hasproperty(ParallelStencil.ParallelKernel, :PKG_KERNELABSTRACTIONS) ? ParallelStencil.ParallelKernel.PKG_KERNELABSTRACTIONS : Symbol(:KernelAbstractions)

TEST_PACKAGES = collect(SUPPORTED_PACKAGES)
if PKG_KERNELABSTRACTIONS ∉ TEST_PACKAGES
    push!(TEST_PACKAGES, PKG_KERNELABSTRACTIONS)
end

@static if PKG_CUDA in TEST_PACKAGES
    import CUDA
    if !CUDA.functional()
        TEST_PACKAGES = filter!(x -> x ≠ PKG_CUDA, TEST_PACKAGES)
    end
end
@static if PKG_AMDGPU in TEST_PACKAGES
    import AMDGPU
    if !AMDGPU.functional()
        TEST_PACKAGES = filter!(x -> x ≠ PKG_AMDGPU, TEST_PACKAGES)
    end
end
@static if PKG_METAL in TEST_PACKAGES
    @static if Sys.isapple()
        import Metal
        if !Metal.functional()
            TEST_PACKAGES = filter!(x -> x ≠ PKG_METAL, TEST_PACKAGES)
        end
    else
        TEST_PACKAGES = filter!(x -> x ≠ PKG_METAL, TEST_PACKAGES)
    end
end
@static if PKG_POLYESTER in TEST_PACKAGES
    import Polyester
end
@static if PKG_KERNELABSTRACTIONS in TEST_PACKAGES
    if Base.find_package("KernelAbstractions") === nothing
        TEST_PACKAGES = filter!(x -> x ≠ PKG_KERNELABSTRACTIONS, TEST_PACKAGES)
    else
        import KernelAbstractions
    end
end

Base.retry_load_extensions()

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

kernelabstractions_handle_cases() = Pair{Symbol,DataType}[]

@static if PKG_KERNELABSTRACTIONS in TEST_PACKAGES
    function kernelabstractions_handle_cases()
        candidates = [
            (:cpu, KernelAbstractions.CPU),
            (:gpu_cuda, KernelAbstractions.CUDABackend),
            (:gpu_amd, KernelAbstractions.ROCBackend),
            (:gpu_metal, KernelAbstractions.MetalBackend),
            (:gpu_oneapi, KernelAbstractions.oneAPIBackend),
        ]
        return [(symbol, handle_type) for (symbol, handle_type) in candidates if isdefined(KernelAbstractions, nameof(handle_type))]
    end
end

is_multi_architecture(package::Symbol) = package == PKG_KERNELABSTRACTIONS

@static for package in TEST_PACKAGES eval(:(
    @testset "$(basename(@__FILE__)) (package: $(nameof($package)))" begin
        @require !@is_initialized()
        @init_parallel_kernel($package, Float32)

        default_symbol = default_hardware_symbol($package)
        @test current_hardware() == default_symbol

        if is_multi_architecture($package)
            cases = kernelabstractions_handle_cases()
            @test !isempty(cases)

            for (symbol, handle_type) in cases
                select_hardware(symbol)
                @test current_hardware() == symbol
                translated = ParallelKernel.handle(symbol)
                @test isa(translated, handle_type)
            end

            select_hardware(:cpu)
            @test current_hardware() == :cpu
            @test_throws ArgumentError select_hardware(:unsupported_hardware_symbol)
            @test_throws ArgumentError ParallelKernel.handle(:unsupported_hardware_symbol)
        else
            invalid_symbol = invalid_hardware_symbol($package, default_symbol)
            @test_throws ArgumentError select_hardware(invalid_symbol)
            @test current_hardware() == default_symbol
            @test_throws ArgumentError ParallelKernel.handle(default_symbol)
        end

        @reset_parallel_kernel()
        @require !@is_initialized()
        @init_parallel_kernel($package, Float32)
        @test current_hardware() == default_symbol
        @reset_parallel_kernel()
    end;
)) end == nothing || true
