using Test
import ParallelStencil
using ParallelStencil.ParallelKernel
import ParallelStencil.ParallelKernel: @reset_parallel_kernel, @is_initialized, @select_hardware, @current_hardware, handle, @ka, SUPPORTED_PACKAGES, PKG_CUDA, PKG_AMDGPU, PKG_METAL, PKG_KERNELABSTRACTIONS, PKG_POLYESTER
import ParallelStencil.ParallelKernel: @require, @prettyexpand, @gorgeousexpand, gorgeousstring, @isgpu
import ParallelStencil.ParallelKernel: checkargs_hide_communication, hide_communication_gpu
using ParallelStencil.ParallelKernel.Exceptions
include(joinpath(@__DIR__, "../packages.jl")) # Selection of the packages to test (TEST_PACKAGES), imports of the backend packages.


@testset "$(basename(@__FILE__))" verbose=true begin # NOTE: the package testsets are nested in this testset (and the package loop is run at run time) in order to test all packages even if the tests of one package fail.
for package in TEST_PACKAGES
    FloatDefault = (package == PKG_METAL) ? Float32 : Float64 # Metal does not support Float64
    
eval(:(
    @testset "$(basename(@__FILE__)) (package: $(nameof($package)))" begin
        @testset "1. hide_communication macro" begin
            @require !@is_initialized()
            @init_parallel_kernel($package, $FloatDefault)
            @require @is_initialized()
            @testset "@hide_communication boundary_width block (macro expansion)" begin
                @static if @isgpu($package)
                    block = string(@gorgeousexpand(1, @hide_communication(boundary_width, begin @parallel f(A, B); communication(); end)))
                    @test occursin("ranges_outer = ParallelStencil.ParallelKernel.get_ranges_outer(boundary_width, ParallelStencil.ParallelKernel.get_ranges(A, B))", block)
                    @test occursin("ranges_inner = ParallelStencil.ParallelKernel.get_ranges_inner(boundary_width, ParallelStencil.ParallelKernel.get_ranges(A, B))", block)
                    @test occursin("@hide_communication ranges_outer ranges_inner computation_calls = 1 begin\n            @parallel f(A, B)\n            communication()\n        end\nend", block)

                    block = string(@gorgeousexpand(1, @hide_communication(boundary_width, begin @parallel ranges f(A, B); communication(); end)))
                    @test occursin("ranges_outer = ParallelStencil.ParallelKernel.get_ranges_outer(boundary_width, ranges)", block)
                    @test occursin("ranges_inner = ParallelStencil.ParallelKernel.get_ranges_inner(boundary_width, ranges)", block)
                    @test occursin("@hide_communication ranges_outer ranges_inner computation_calls = 1 begin\n            @parallel f(A, B)\n            communication()\n        end\nend", block)

                    block = string(@gorgeousexpand(1, @hide_communication(boundary_width, computation_calls=2, begin @parallel f(A, B); @parallel g(B, C); communication(); end)))
                    @test occursin("ranges_outer = ParallelStencil.ParallelKernel.get_ranges_outer(boundary_width, ParallelStencil.ParallelKernel.get_ranges(A, B), ParallelStencil.ParallelKernel.get_ranges(B, C))", block)
                    @test occursin("ranges_inner = ParallelStencil.ParallelKernel.get_ranges_inner(boundary_width, ParallelStencil.ParallelKernel.get_ranges(A, B), ParallelStencil.ParallelKernel.get_ranges(B, C))", block)
                    @test occursin("@hide_communication ranges_outer ranges_inner computation_calls = 2 begin\n            @parallel f(A, B)\n            @parallel g(B, C)\n            communication()\n        end\nend", block)

                    block = string(@gorgeousexpand(1, @hide_communication(boundary_width, computation_calls=2, begin @parallel ranges1 f(A, B); @parallel g(B, C); communication(); end)))
                    @test occursin("ranges_outer = ParallelStencil.ParallelKernel.get_ranges_outer(boundary_width, ranges1, ParallelStencil.ParallelKernel.get_ranges(B, C))", block)
                    @test occursin("ranges_inner = ParallelStencil.ParallelKernel.get_ranges_inner(boundary_width, ranges1, ParallelStencil.ParallelKernel.get_ranges(B, C))", block)
                    @test occursin("@hide_communication ranges_outer ranges_inner computation_calls = 2 begin\n            @parallel f(A, B)\n            @parallel g(B, C)\n            communication()\n        end\nend", block)

                    block = string(@gorgeousexpand(1, @hide_communication(boundary_width, computation_calls=2, begin @parallel ranges1 f(A, B); @parallel ranges2 g(B, C); communication(); end)))
                    @test occursin("ranges_outer = ParallelStencil.ParallelKernel.get_ranges_outer(boundary_width, ranges1, ranges2)", block)
                    @test occursin("ranges_inner = ParallelStencil.ParallelKernel.get_ranges_inner(boundary_width, ranges1, ranges2)", block)
                    @test occursin("@hide_communication ranges_outer ranges_inner computation_calls = 2 begin\n            @parallel f(A, B)\n            @parallel g(B, C)\n            communication()\n        end\nend", block)
                end;
            end;
            @testset "@hide_communication" begin
                @parallel_indices (ix,iy,iz) function add_indices!(A)
                    A[ix,iy,iz] = A[ix,iy,iz] + ix + (iy-1)*size(A,1) + (iz-1)*size(A,1)*size(A,2); # NOTE: $ix, $iy, $iz come from ParallelStencil.INDICES.
                    return
                end
                function communication!(A)
                    A[  1,  :,  :] .= A[end-1,    :,    :]
                    A[end,  :,  :] .= A[    2,    :,    :]
                    A[  :,  1,  :] .= A[    :,end-1,    :]
                    A[  :,end,  :] .= A[    :,    2,    :]
                    A[  :,  :,  1] .= A[    :,    :,end-1]
                    A[  :,  :,end] .= A[    :,    :,    2]
                    return A
                end
                function communication_y!(A)
                    A[  :,  1,  :] .= A[    :,end-1,    :]
                    A[  :,end,  :] .= A[    :,    2,    :]
                    return A
                end
                function communication_z!(A)
                    A[  :,  :,  1] .= A[    :,    :,end-1]
                    A[  :,  :,end] .= A[    :,    :,    2]
                    return A
                end
                add_indices2! = add_indices!
                add_indices3! = add_indices!
                @static if $package == $PKG_KERNELABSTRACTIONS
                    @testset "KernelAbstractions runtime reselection" begin
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
                        try
                            for symbol in valid_symbols
                                @select_hardware(symbol)
                                @require @current_hardware() == symbol
                                FloatHardware = (symbol == :gpu_metal) ? Float32 : Float64 # Metal does not support Float64
                                A  = @zeros(6, 7, 8, eltype=FloatHardware)
                                @hide_communication (2,2,3) begin
                                    @parallel add_indices!(A);
                                    communication!(A);
                                end
                                @test all(Array(A) .== communication!([ix + (iy-1)*size(A,1) + (iz-1)*size(A,1)*size(A,2) for ix=1:size(A,1), iy=1:size(A,2), iz=1:size(A,3)]))
                            end
                        finally
                            @select_hardware(:cpu) # NOTE: restore the default hardware also if a test errors, as the hardware selection persists for all remaining tests of this file.
                        end
                    end
                end
                @testset "@hide_communication boundary_width block" begin
                    A  = @zeros(6, 7, 8)
                    @hide_communication (2,2,3) begin
                        @parallel add_indices!(A);
                        communication!(A);
                    end
                    @test all(Array(A) .== communication!([ix + (iy-1)*size(A,1) + (iz-1)*size(A,1)*size(A,2) for ix=1:size(A,1), iy=1:size(A,2), iz=1:size(A,3)]))
                end;
                @testset "@hide_communication boundary_width block" begin  # This test verifies that the results are correct, even for CUDA.jl < v2.0, where it cannot overlap.
                    A  = @zeros(6, 7, 8)
                    @hide_communication (2,2,3) begin
                        @parallel add_indices!(A);
                        A[  1,  :,  :] .= A[end-1,    :,    :]
                        if true
                            A[end,  :,  :] .= A[    2,    :,    :]
                        end
                        communication_y!(A);
                        communication_z!(A);
                    end
                    @test all(Array(A) .== communication!([ix + (iy-1)*size(A,1) + (iz-1)*size(A,1)*size(A,2) for ix=1:size(A,1), iy=1:size(A,2), iz=1:size(A,3)]))
                end;
                @testset "@hide_communication boundary_width block" begin
                    A  = @zeros(6, 7, 8)
                    @parallel_indices (iy,iz) bc_xl(A) = (A[        1,iy,iz]=A[size(A,1)-1,iy,iz]; return) # NOTE: using size(A,1) instead of `end` due to a limitation of Polyester
                    @parallel_indices (iy,iz) bc_xr(A) = (A[size(A,1),iy,iz]=A[          2,iy,iz]; return) # ...
                    @hide_communication (2,2,3) begin
                        @parallel add_indices!(A);
                        @parallel (1:size(A,2), 1:size(A,3)) bc_xl(A);
                        if true
                            @parallel (1:size(A,2), 1:size(A,3)) bc_xr(A);
                        end
                        communication_y!(A);
                        communication_z!(A);
                    end
                    @test all(Array(A) .== communication!([ix + (iy-1)*size(A,1) + (iz-1)*size(A,1)*size(A,2) for ix=1:size(A,1), iy=1:size(A,2), iz=1:size(A,3)]))
                end;
                @testset "@hide_communication boundary_width computation_calls=2 block" begin
                    A  = @zeros(6, 7, 8)
                    @hide_communication (2,2,3) computation_calls=2 begin
                        @parallel add_indices!(A);
                        @parallel add_indices2!(A);
                        communication!(A);
                    end
                    @test all(Array(A) .== communication!([2*(ix + (iy-1)*size(A,1) + (iz-1)*size(A,1)*size(A,2)) for ix=1:size(A,1), iy=1:size(A,2), iz=1:size(A,3)]))
                end;
                @testset "@hide_communication boundary_width computation_calls=2 block" begin
                    A  = @zeros(6, 7, 8)
                    @hide_communication (2,2,3) computation_calls=2 begin
                        @parallel add_indices!(A);
                        @parallel (1:6, 1:7, 1:8) add_indices2!(A);
                        communication!(A);
                    end
                    @test all(Array(A) .== communication!([2*(ix + (iy-1)*size(A,1) + (iz-1)*size(A,1)*size(A,2)) for ix=1:size(A,1), iy=1:size(A,2), iz=1:size(A,3)]))
                end;
                @testset "@hide_communication boundary_width computation_calls=2 block" begin
                    A  = @zeros(6, 7, 8)
                    @hide_communication (2,2,3) computation_calls=2 begin
                        @parallel (1:6, 1:7, 1:8) add_indices!(A);
                        @parallel (1:6, 1:7, 1:8) add_indices2!(A);
                        communication!(A);
                    end
                    @test all(Array(A) .== communication!([2*(ix + (iy-1)*size(A,1) + (iz-1)*size(A,1)*size(A,2)) for ix=1:size(A,1), iy=1:size(A,2), iz=1:size(A,3)]))
                end;
                @testset "@hide_communication boundary_width computation_calls=3 block" begin
                    A  = @zeros(6, 7, 8)
                    @hide_communication (2,2,3) computation_calls=3 begin
                        @parallel add_indices!(A);
                        @parallel (1:6, 1:7, 1:8) add_indices2!(A);
                        @parallel add_indices3!(A);
                        communication!(A);
                    end
                    @test all(Array(A) .== communication!([3*(ix + (iy-1)*size(A,1) + (iz-1)*size(A,1)*size(A,2)) for ix=1:size(A,1), iy=1:size(A,2), iz=1:size(A,3)]))
                end;
                @testset "@hide_communication ranges_outer ranges_inner block" begin
                    A  = @zeros(6, 7, 8)
                    ranges_outer = ParallelStencil.ParallelKernel.get_ranges_outer((2, 2, 3), ParallelStencil.ParallelKernel.get_ranges(A))
                    ranges_inner = ParallelStencil.ParallelKernel.get_ranges_inner((2, 2, 3), ParallelStencil.ParallelKernel.get_ranges(A))
                    @hide_communication ranges_outer ranges_inner begin
                        @parallel add_indices!(A);
                        communication!(A);
                    end
                    @test all(Array(A) .== communication!([ix + (iy-1)*size(A,1) + (iz-1)*size(A,1)*size(A,2) for ix=1:size(A,1), iy=1:size(A,2), iz=1:size(A,3)]))
                end;
                @testset "@hide_communication: implicit synchronization of the arrays suspended" begin # NOTE: CUDA.jl (from v6.3) and AMDGPU.jl order every use of an array on a stream after its last use on another stream; unless this is suspended, the communication waits for the inner computation.
                    @static if $(package in (PKG_CUDA, PKG_AMDGPU))
                        @static if $(package == PKG_CUDA)
                            has_optout       = isdefined(CUDA, :enable_synchronization!)
                            has_query        = has_optout
                            isdone(s)        = CUDA.isdone(s)
                            sync_enabled(A)  = A.data[].synchronizing
                            set_sync!(A, on) = CUDA.enable_synchronization!(A, on)
                        else
                            has_optout       = isdefined(AMDGPU, :enable_synchronization!)
                            has_query        = isdefined(AMDGPU, :synchronization_enabled)
                            isdone(s)        = AMDGPU.HIP.isdone(s)
                            sync_enabled(A)  = AMDGPU.synchronization_enabled(A)
                            set_sync!(A, on) = AMDGPU.enable_synchronization!(A, on)
                        end
                        @parallel_indices (ix,iy,iz) function slow!(A, n)
                            x = A[ix,iy,iz]
                            for _ in 1:n
                                x = x*0.999 + 0.001
                            end
                            A[ix,iy,iz] = x
                            return
                        end
                        @parallel_indices (ix,iy,iz) function touch!(A)
                            A[ix,iy,iz] = A[ix,iy,iz]
                            return
                        end
                        function communication_busy!(A, busy) # NOTE: as update_halo! of ImplicitGlobalGrid: a kernel on the array and a memory copy through its pointer, on a stream of its own.
                            s = ParallelStencil.ParallelKernel.@get_priority_stream(8)
                            @parallel_async (1:1, 1:1, 1:1) stream=s touch!(A)
                            pointer(A)
                            ParallelStencil.ParallelKernel.@synchronize(s)
                            busy[] = !isdone(ParallelStencil.ParallelKernel.@get_stream(1)) # NOTE: whether the inner computation is still running.
                            return
                        end
                        A    = @zeros(64, 64, 64)
                        busy = Ref(false)
                        n    = 1
                        @hide_communication (8,8,8) begin # NOTE: warm-up (compilation).
                            @parallel slow!(A, n);
                            communication_busy!(A, busy);
                        end
                        while (@elapsed @parallel slow!(A, n)) < 0.2 n *= 2 end # NOTE: an inner computation of at least about 0.2 s.
                        @hide_communication (8,8,8) begin
                            @parallel slow!(A, n);
                            communication_busy!(A, busy);
                        end
                        @static if $(package == PKG_CUDA)
                            @test busy[]
                        else
                            if has_optout @test busy[] else @test_broken busy[] end # NOTE: AMDGPU.jl versions without enable_synchronization! cannot overlap.
                        end
                        if has_optout && has_query # NOTE: the state of each array is restored, also of an array excluded from the implicit synchronization by the user.
                            B = @zeros(64, 64, 64)
                            set_sync!(B, false)
                            @hide_communication (8,8,8) begin
                                @parallel slow!(A, 1);
                                communication_busy!(B, busy);
                            end
                            @test sync_enabled(A)
                            @test !sync_enabled(B)
                            set_sync!(B, true)
                        end
                    end
                end;
                @testset "@get_stream and @get_priority_stream" begin
                    @static if ParallelStencil.ParallelKernel.iscpu($package)
                        @test ParallelStencil.ParallelKernel.@get_stream(1) === nothing
                        @test ParallelStencil.ParallelKernel.@get_priority_stream(1) === nothing
                    end
                end;
            end;
            @reset_parallel_kernel()
        end;
        @testset "2. Exceptions" begin
            @require !@is_initialized()
            @init_parallel_kernel($package, $FloatDefault)
            @require @is_initialized
            @static if $package == $PKG_KERNELABSTRACTIONS
                @require @current_hardware() == :cpu
            end
            @testset "arguments @hide_communication" begin
                @test_throws ArgumentError checkargs_hide_communication(:boundary_width, :block)               # Error: the last argument must be a code block.
                @test_throws ArgumentError checkargs_hide_communication(:ranges_outer, :ranges_inner, :block)  # Error: the last argument must be a code block.
                @static if @isgpu($package)
                    @test_throws ArgumentError hide_communication_gpu(:boundary_width, :(@parallel f()))                                                                      # Error: missing (bc and) communication code.
                    @test_throws ArgumentError hide_communication_gpu(:boundary_width, :(communication()))                                                                    # Error: missing @parallel call.
                    @test_throws ArgumentError hide_communication_gpu(:boundary_width, :(begin @parallel f(); end))                                                           # Error: missing (bc and) communication code.
                    @test_throws ArgumentError hide_communication_gpu(:boundary_width, :(begin communication(); end))                                                         # Error: missing @parallel call.
                    @test_throws KeywordArgumentError hide_communication_gpu(:boundary_width, :(begin @parallel ranges g(A, B); communication(); end); computation_calls=0)          # Error: computation_calls<1.
                    block = :(begin do_something(); @parallel f(); communication(); end);          @test_throws ArgumentError hide_communication_gpu(:boundary_width, block)  # Error: block does not start with @parallel call
                    block = :(begin @parallel stream=mystream f(); communication(); end);          @test_throws ArgumentError hide_communication_gpu(:boundary_width, block)  # Error: invalid keyword argument 'stream'.
                    block = :(begin @parallel nblocks nthreads f(); communication(); end);         @test_throws ArgumentError hide_communication_gpu(:boundary_width, block)  # Error: invalid optional arguments 'nblocks' and 'nthreads'.
                    block = :(begin @parallel ranges nblocks nthreads f(); communication(); end);  @test_throws ArgumentError hide_communication_gpu(:boundary_width, block)  # Error: invalid optional arguments 'ranges', 'nblocks' and 'nthreads'.

                    @test_throws ArgumentError hide_communication_gpu(:ranges_outer, :ranges_inner, :(@parallel f()))                                                                      # Error: missing (bc and) communication code.
                    @test_throws ArgumentError hide_communication_gpu(:ranges_outer, :ranges_inner, :(communication()))                                                                    # Error: missing @parallel call.
                    @test_throws ArgumentError hide_communication_gpu(:ranges_outer, :ranges_inner, :(begin @parallel f(); end))                                                           # Error: missing (bc and) communication code.
                    @test_throws ArgumentError hide_communication_gpu(:ranges_outer, :ranges_inner, :(begin communication(); end))                                                         # Error: missing @parallel call.
                    block = :(begin do_something(); @parallel f(); communication(); end);          @test_throws ArgumentError hide_communication_gpu(:ranges_outer, :ranges_inner, block)  # Error: block does not start with @parallel call
                    block = :(begin @parallel stream=mystream f(); communication(); end);          @test_throws ArgumentError hide_communication_gpu(:ranges_outer, :ranges_inner, block)  # Error: invalid keyword argument 'stream'.
                    block = :(begin @parallel ranges f(); communication(); end);                   @test_throws ArgumentError hide_communication_gpu(:ranges_outer, :ranges_inner, block)  # Error: invalid optional arguments 'ranges'.
                    block = :(begin @parallel nblocks nthreads f(); communication(); end);         @test_throws ArgumentError hide_communication_gpu(:ranges_outer, :ranges_inner, block)  # Error: invalid optional arguments 'nblocks' and 'nthreads'.
                    block = :(begin @parallel ranges nblocks nthreads f(); communication(); end);  @test_throws ArgumentError hide_communication_gpu(:ranges_outer, :ranges_inner, block)  # Error: invalid optional arguments 'ranges', 'nblocks' and 'nthreads'.

                    block = :(begin @parallel f(); @parallel g(); communication(); end);                          @test_throws ArgumentError hide_communication_gpu(:ranges_outer, :ranges_inner, block)  # Error: missing arguments 'ranges' in @parallel call for bc computations.
                    block = :(begin @parallel f(); @parallel stream=mystream g(); communication(); end);          @test_throws ArgumentError hide_communication_gpu(:ranges_outer, :ranges_inner, block)  # Error: invalid keyword argument in @parallel call for bc computations.
                    block = :(begin @parallel f(); @parallel nblocks nthreads g(); communication(); end);         @test_throws ArgumentError hide_communication_gpu(:ranges_outer, :ranges_inner, block)  # Error: invalid optional arguments 'nblocks' and 'nthreads' in @parallel call for bc computations.
                    block = :(begin @parallel f(); @parallel ranges nblocks nthreads g(); communication(); end);  @test_throws ArgumentError hide_communication_gpu(:ranges_outer, :ranges_inner, block)  # Error: invalid optional arguments 'ranges', 'nblocks' and 'nthreads' in @parallel call for bc computations.
                end
            end;
            @testset "@hide_communication ranges determination when computation_calls>1" begin
                A  = @zeros(6, 7, 8)
                @test_throws ArgumentError ParallelStencil.ParallelKernel.get_ranges_outer((2,2,3), ParallelStencil.ParallelKernel.get_ranges(A), (1:6, 1:9, 1:8), ParallelStencil.ParallelKernel.get_ranges(A))  # Error: the ranges of the computation calls are not all equal.
                @test_throws ArgumentError ParallelStencil.ParallelKernel.get_ranges_inner((2,2,3), ParallelStencil.ParallelKernel.get_ranges(A), (1:6, 1:9, 1:8), ParallelStencil.ParallelKernel.get_ranges(A))  # Error: the ranges of the computation calls are not all equal.
            end;
            @reset_parallel_kernel()
        end;
    end;
))

end
end;
