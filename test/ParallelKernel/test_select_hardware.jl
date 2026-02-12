using Test
using ParallelStencil.ParallelKernel
import ParallelStencil.ParallelKernel: @reset_parallel_kernel, @init_parallel_kernel, @is_initialized, @select_hardware, @current_hardware, @get_hardware, handle, SUPPORTED_PACKAGES, PKG_CUDA, PKG_AMDGPU, PKG_METAL, PKG_THREADS, PKG_POLYESTER, PKG_KERNELABSTRACTIONS
import ParallelStencil.ParallelKernel: @require, @prettystring
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
			@reset_parallel_kernel()
			@require !@is_initialized()
			@testset "Runtime hardware (re-)selection" begin
				package_symbol = $(QuoteNode(package))
				@init_parallel_kernel($package, Float64)
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

				valid_symbols = Symbol[]
				if $package == $PKG_KERNELABSTRACTIONS
					valid_symbols = [:cpu, :gpu_cuda, :gpu_amd, :gpu_metal, :gpu_oneapi]
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
				end
				last_selected = last(valid_symbols)
				@select_hardware(last_selected)
				@test @current_hardware() == last_selected

				if $package == $PKG_KERNELABSTRACTIONS
					handle_symbols = [:cpu]
					if isdefined(KernelAbstractions, :CUDABackend) push!(handle_symbols, :gpu_cuda) end
					if isdefined(KernelAbstractions, :ROCBackend) push!(handle_symbols, :gpu_amd) end
					if isdefined(KernelAbstractions, :MetalBackend) push!(handle_symbols, :gpu_metal) end
					if isdefined(KernelAbstractions, :oneAPIBackend) push!(handle_symbols, :gpu_oneapi) end
					for symbol in handle_symbols
						@testset "handle($(symbol))" begin
							if symbol == :cpu
								@test handle(symbol, package_symbol) == CPU()
							elseif symbol == :gpu_cuda
								@test handle(symbol, package_symbol) == CUDABackend()
							elseif symbol == :gpu_amd
								@test handle(symbol, package_symbol) == ROCBackend()
							elseif symbol == :gpu_metal
								@test handle(symbol, package_symbol) == MetalBackend()
							elseif symbol == :gpu_oneapi
								@test handle(symbol, package_symbol) == oneAPIBackend()
							end
						end
					end
				else
					@test_throws ArgumentError handle(:cpu, package_symbol)
				end

				all_symbols = (:cpu, :gpu_cuda, :gpu_amd, :gpu_metal, :gpu_oneapi)
				invalid_symbols = filter(s -> !(s in valid_symbols), all_symbols)
				last_valid = @current_hardware()
				for symbol in invalid_symbols
					@test_throws ArgumentError @select_hardware(symbol)
					@test @current_hardware() == last_valid
				end
			end
			@reset_parallel_kernel()
	end;
)) end == nothing || true;
