using Test
using ParallelStencil
import ParallelStencil: @reset_parallel_stencil, @init_parallel_stencil, @is_initialized, @select_hardware, @current_hardware, @get_hardware, SUPPORTED_PACKAGES, PKG_CUDA, PKG_AMDGPU, PKG_METAL, PKG_THREADS, PKG_POLYESTER, PKG_KERNELABSTRACTIONS
import ParallelStencil: @require, @symbols
import ParallelStencil.ParallelKernel: handle
import KernelAbstractions: CPU # NOTE: the GPU backend types (e.g. CUDABackend) are defined by the extensions of KernelAbstractions, i.e. only if the corresponding GPU package is loaded; they are referred to qualified below.

include(joinpath(@__DIR__, "packages.jl")) # Selection of the packages to test (TEST_PACKAGES), imports of the backend packages.



@testset "$(basename(@__FILE__))" verbose=true begin # NOTE: the package testsets are nested in this testset (and the package loop is run at run time) in order to test all packages even if the tests of one package fail.
for package in TEST_PACKAGES  eval(:(
	@testset "$(basename(@__FILE__)) (package: $(nameof($package)))" begin
			@reset_parallel_stencil()
			@require !@is_initialized()
			@testset "Runtime hardware (re-)selection wrappers" begin
				package_symbol = $(QuoteNode(package))
				@init_parallel_stencil(package = $package, numbertype = Float64, ndims = 3)
				facade_hw = @get_hardware()
				kernel_hw = ParallelStencil.ParallelKernel.@get_hardware()
				@test facade_hw == kernel_hw
				@static if $package == $PKG_KERNELABSTRACTIONS
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
				@static if $package == $PKG_KERNELABSTRACTIONS
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
					@test ParallelStencil.ParallelKernel.@current_hardware() == symbol
					@require @is_initialized()
					@test @is_initialized()
					@static if $package == $PKG_KERNELABSTRACTIONS
						@test length(@symbols($(@__MODULE__), Data)) <= 1
						@test length(@symbols($(@__MODULE__), TData)) <= 1
					else
						@test @isdefined(Data)
						@test @isdefined(TData)
					end
				end

				@static if $package == $PKG_KERNELABSTRACTIONS
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
								@test handle(symbol, package_symbol) == KernelAbstractions.CUDABackend()
							elseif symbol == :gpu_amd
								@test handle(symbol, package_symbol) == KernelAbstractions.ROCBackend()
							elseif symbol == :gpu_metal
								@test handle(symbol, package_symbol) == KernelAbstractions.MetalBackend()
							elseif symbol == :gpu_oneapi
								@test handle(symbol, package_symbol) == KernelAbstractions.oneAPIBackend()
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
)) end
end;
