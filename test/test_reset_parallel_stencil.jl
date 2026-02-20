using Test
using ParallelStencil
import ParallelStencil: @reset_parallel_stencil, @is_initialized, @get_package, @get_numbertype, @get_hardware, @get_ndims, @select_hardware, @current_hardware, SUPPORTED_PACKAGES, PKG_CUDA, PKG_AMDGPU, PKG_METAL, PKG_KERNELABSTRACTIONS, PKG_POLYESTER, PKG_NONE, NUMBERTYPE_NONE, NDIMS_NONE
import ParallelStencil.ParallelKernel: @isdefined_at_pt
import ParallelStencil: @require, @symbols
TEST_PACKAGES = SUPPORTED_PACKAGES
@static if PKG_CUDA in TEST_PACKAGES
    import CUDA
    if !CUDA.functional() TEST_PACKAGES = filter!(x->x≠PKG_CUDA, TEST_PACKAGES) end
end
@static if PKG_AMDGPU in TEST_PACKAGES
    import AMDGPU
    if !AMDGPU.functional() TEST_PACKAGES = filter!(x->x≠PKG_AMDGPU, TEST_PACKAGES) end
end
@static if PKG_KERNELABSTRACTIONS in TEST_PACKAGES
    import KernelAbstractions
    if !KernelAbstractions.functional(KernelAbstractions.CPU()) TEST_PACKAGES = filter!(x->x≠PKG_KERNELABSTRACTIONS, TEST_PACKAGES) end
end
@static if PKG_METAL in TEST_PACKAGES
    import Metal
    if !Metal.functional() TEST_PACKAGES = filter!(x->x≠PKG_METAL, TEST_PACKAGES) end
end
@static if PKG_POLYESTER in TEST_PACKAGES
    import Polyester
end
Base.retry_load_extensions() # Potentially needed to load the extensions after the packages have been filtered.


@static for package in TEST_PACKAGES  eval(:(
    @testset "$(basename(@__FILE__)) (package: $(nameof($package)))" begin
        @testset "1. Reset of ParallelStencil" begin
            @testset "Reset if not initialized" begin
                @require !@is_initialized()
                hw_before_reset = @get_hardware()
                @reset_parallel_stencil()
                @test @get_hardware() == hw_before_reset
                @test !@is_initialized()
                @test @get_package() == $PKG_NONE
                @test @get_numbertype() == $NUMBERTYPE_NONE
                @test @get_ndims() == $NDIMS_NONE
            end;
            @testset "Reset if initialized" begin
                @require !@is_initialized()
                @init_parallel_stencil($package, Float64, 3)
                @require @is_initialized() && @get_package() == $package
                @static if $package == $PKG_KERNELABSTRACTIONS
                    valid_symbols = Symbol[:cpu]
                    @static if PKG_CUDA in TEST_PACKAGES
                        push!(valid_symbols, :gpu_cuda)
                    end
                    @static if PKG_AMDGPU in TEST_PACKAGES
                        push!(valid_symbols, :gpu_amd)
                    end
                    @static if PKG_METAL in TEST_PACKAGES
                        push!(valid_symbols, :gpu_metal)
                    end
                    @static if isdefined(ParallelStencil.ParallelKernel, :PKG_ONEAPI) && ParallelStencil.ParallelKernel.PKG_ONEAPI in TEST_PACKAGES
                        push!(valid_symbols, :gpu_oneapi)
                    end
                    for symbol in valid_symbols
                        @select_hardware(symbol)
                        @require @current_hardware() == symbol
                    end
                    hw_before_reset = @get_hardware()
                    @reset_parallel_stencil()
                    @test @get_hardware() == hw_before_reset
                    @test !@isdefined_at_pt(Data.Device) # KernelAbstractions intentionally lacks convenience modules.
                else
                    @reset_parallel_stencil()
                    @test length(@symbols($(@__MODULE__), Data)) == 1
                end
                @test !@is_initialized()
                @test @get_package() == $PKG_NONE
                @test @get_numbertype() == $NUMBERTYPE_NONE
                @test @get_ndims() == $NDIMS_NONE
            end;
        end;
    end;
)) end == nothing || true;
