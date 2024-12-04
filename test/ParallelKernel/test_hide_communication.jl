using Test
import ParallelStencil
using ParallelStencil.ParallelKernel
import ParallelStencil.ParallelKernel: @reset_parallel_kernel, @is_initialized, SUPPORTED_PACKAGES, PKG_CUDA, PKG_AMDGPU, PKG_METAL, PKG_POLYESTER
import ParallelStencil.ParallelKernel: @require, @prettyexpand, @gorgeousexpand, gorgeousstring, @isgpu
import ParallelStencil.ParallelKernel: checkargs_hide_communication, hide_communication_gpu
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
Base.retry_load_extensions() # Potentially needed to load the extensions after the packages have been filtered.


@static for package in TEST_PACKAGES
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
                    ranges_outer = ParallelStencil.ParallelKernel.get_ranges_outer((1, 1, 2), ParallelStencil.ParallelKernel.get_ranges(A))
                    ranges_inner = ParallelStencil.ParallelKernel.get_ranges_inner((1, 1, 2), ParallelStencil.ParallelKernel.get_ranges(A))
                    @hide_communication ranges_outer ranges_inner begin
                        @parallel add_indices!(A);
                        communication!(A);
                    end
                    @test all(Array(A) .== communication!([ix + (iy-1)*size(A,1) + (iz-1)*size(A,1)*size(A,2) for ix=1:size(A,1), iy=1:size(A,2), iz=1:size(A,3)]))
                end;
            end;
            @reset_parallel_kernel()
        end;
        @testset "2. Exceptions" begin
            @require !@is_initialized()
            @init_parallel_kernel($package, $FloatDefault)
            @require @is_initialized
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

end == nothing || true;
