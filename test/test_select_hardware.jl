using Test
using ParallelStencil
import ParallelStencil: @reset_parallel_stencil, @init_parallel_stencil, @is_initialized, @select_hardware, @current_hardware, @get_hardware, SUPPORTED_PACKAGES, PKG_CUDA, PKG_AMDGPU, PKG_METAL, PKG_THREADS, PKG_POLYESTER, PKG_KERNELABSTRACTIONS
import ParallelStencil: @require, @symbols
import ParallelStencil.ParallelKernel: handle
@static if PKG_KERNELABSTRACTIONS in SUPPORTED_PACKAGES
    import KernelAbstractions: CPU, CUDABackend, ROCBackend, MetalBackend, oneAPIBackend
end

TEST_PACKAGES = SUPPORTED_PACKAGES

@static if PKG_CUDA in TEST_PACKAGES
    import CUDA
    if !CUDA.functional() TEST_PACKAGES = filter!(x -> x ≠ PKG_CUDA, TEST_PACKAGES) end
end
@static if PKG_AMDGPU in TEST_PACKAGES
    import AMDGPU
    if !AMDGPU.functional() TEST_PACKAGES = filter!(x -> x ≠ PKG_AMDGPU, TEST_PACKAGES) end
end
@static if PKG_METAL in TEST_PACKAGES
	import Metal
	if !Metal.functional() TEST_PACKAGES = filter!(x->x≠PKG_METAL, TEST_PACKAGES) end
end
@static if PKG_POLYESTER in TEST_PACKAGES
    import Polyester
end
@static if PKG_KERNELABSTRACTIONS in TEST_PACKAGES
    import KernelAbstractions
    if !KernelAbstractions.functional(KernelAbstractions.CPU()) TEST_PACKAGES = filter!(x -> x ≠ PKG_KERNELABSTRACTIONS, TEST_PACKAGES) end
end
Base.retry_load_extensions()


@static for package in TEST_PACKAGES  eval(:(
	@testset "$(basename(@__FILE__)) (package: $(nameof($package)))" begin
			@reset_parallel_stencil()
			@require !@is_initialized()
			@testset "Runtime hardware (re-)selection wrappers" begin
				@init_parallel_stencil(package = $package, numbertype = Float64, ndims = 3)
				facade_hw = @get_hardware()
				kernel_hw = ParallelStencil.ParallelKernel.@get_hardware()
				@test facade_hw == kernel_hw
				if $package == $PKG_KERNELABSTRACTIONS
					@test facade_hw == :cpu
				elseif $package == $PKG_CUDA
					@test facade_hw == :gpu_cuda
				elseif $package == $PKG_AMDGPU
					@test facade_hw == :gpu_amd
				elseif $package == $PKG_METAL
					@test facade_hw == :gpu_metal
				else
					@test facade_hw == :cpu
				end

				valid_symbols = Symbol[]
				if $package == $PKG_KERNELABSTRACTIONS
					push!(valid_symbols, :cpu)
					if PKG_CUDA in TEST_PACKAGES
						@require PKG_CUDA in TEST_PACKAGES
						push!(valid_symbols, :gpu_cuda)
					end
					if PKG_AMDGPU in TEST_PACKAGES
						@require PKG_AMDGPU in TEST_PACKAGES
						push!(valid_symbols, :gpu_amd)
					end
					if PKG_METAL in TEST_PACKAGES
						@require PKG_METAL in TEST_PACKAGES
						push!(valid_symbols, :gpu_metal)
					end
					push!(valid_symbols, :gpu_oneapi)
				elseif $package == $PKG_CUDA
					valid_symbols = [:gpu_cuda]
				elseif $package == $PKG_AMDGPU
					valid_symbols = [:gpu_amd]
				elseif $package == $PKG_METAL
					valid_symbols = [:gpu_metal]
				else
					valid_symbols = [:cpu]
				end

				for symbol in valid_symbols
					@select_hardware(symbol)
					@test @current_hardware() == symbol
					@test ParallelStencil.ParallelKernel.@current_hardware() == symbol
					@require @is_initialized()
					@test @is_initialized()
					@test !isdefined(Data)
					@test !isdefined(TData)
				end

				if $package == $PKG_KERNELABSTRACTIONS
					for symbol in valid_symbols
						@testset "handle($(symbol))" begin
							if symbol == :cpu
								@test handle(symbol) == CPU()
							elseif symbol == :gpu_cuda
								@test handle(symbol) == CUDABackend()
							elseif symbol == :gpu_amd
								@test handle(symbol) == ROCBackend()
							elseif symbol == :gpu_metal
								@test handle(symbol) == MetalBackend()
							elseif symbol == :gpu_oneapi
								@test handle(symbol) == oneAPIBackend()
							end
						end
					end
				end

				all_symbols = (:cpu, :gpu_cuda, :gpu_amd, :gpu_metal, :gpu_oneapi)
				invalid_symbols = filter(s -> !(s in valid_symbols), all_symbols)
				last_valid = @current_hardware()
				for symbol in invalid_symbols
					@test_throws ArgumentError @select_hardware(symbol)
					@test_throws ArgumentError ParallelStencil.ParallelKernel.@select_hardware(symbol)
					@test @current_hardware() == last_valid
					@test ParallelStencil.ParallelKernel.@current_hardware() == last_valid
				end
			end
			@reset_parallel_stencil()
	end;
)) end == nothing || true;
