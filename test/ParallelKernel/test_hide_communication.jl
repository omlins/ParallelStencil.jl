using Test
import ParallelStencil
using ParallelStencil.ParallelKernel
import ParallelStencil.ParallelKernel: @reset_parallel_kernel, @is_initialized, SUPPORTED_PACKAGES, PKG_CUDA, PKG_THREADS
import ParallelStencil.ParallelKernel: @require, longnameof, @prettyexpand, prettystring, @gorgeousexpand, gorgeousstring
import ParallelStencil.ParallelKernel: checkargs_hide_communication, hide_communication, hide_communication_cuda
using ParallelStencil.ParallelKernel.Exceptions
TEST_PACKAGES = SUPPORTED_PACKAGES
@static if PKG_CUDA in TEST_PACKAGES
    import CUDA
    if !CUDA.functional() TEST_PACKAGES = filter!(x->x≠PKG_CUDA, TEST_PACKAGES) end
end

@static for package in TEST_PACKAGES  eval(:(
    @testset "$(basename(@__FILE__)) (package: $(nameof($package)))" begin
        @testset "1. hide_communication macro" begin
            @init_parallel_kernel($package, Float64)
            @require @is_initialized()
            @testset "@hide_communication boundary_width block (macro expansion)" begin
                @static if $package == $PKG_CUDA
                    block = string(@gorgeousexpand(1, @hide_communication(boundary_width, begin @parallel f(A, B); communication(); end)))
                    @test occursin("ranges_outer = ParallelStencil.ParallelKernel.get_ranges_outer(boundary_width, ParallelStencil.ParallelKernel.get_ranges(A, B))", block)
                    @test occursin("ranges_inner = ParallelStencil.ParallelKernel.get_ranges_inner(boundary_width, ParallelStencil.ParallelKernel.get_ranges(A, B))", block)
                    @test occursin("@hide_communication ranges_outer ranges_inner begin\n            @parallel f(A, B)\n            communication()\n        end\nend", block)
                end;
            end;
            @testset "@hide_communication" begin
                @require @is_initialized()
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
                @testset "@hide_communication boundary_width block" begin
                    A  = @zeros(6, 7, 8)
                    @hide_communication (2,2,3) begin
                        @parallel add_indices!(A);
                        communication!(A);
                    end
                    @test all(Array(A) .== communication!([ix + (iy-1)*size(A,1) + (iz-1)*size(A,1)*size(A,2) for ix=1:size(A,1), iy=1:size(A,2), iz=1:size(A,3)]))
                end;
                @testset "@hide_communication boundary_width block" begin  # This test verifies that the results are correct even though in the current version (using CUDA.jl < v2.0), it cannot overlap
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
                    @parallel_indices (iy,iz) bc_xl(A) = (A[  1,iy,iz]=A[end-1,iy,iz]; return)
                    @parallel_indices (iy,iz) bc_xr(A) = (A[end,iy,iz]=A[    2,iy,iz]; return)
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
            @init_parallel_kernel($package, Float64)
            @require @is_initialized
            @testset "arguments @hide_communication" begin
                @test_throws ArgumentError checkargs_hide_communication(:boundary_width, :block)               # Error: the last argument must be a code block.
                @test_throws ArgumentError checkargs_hide_communication(:ranges_outer, :ranges_inner, :block)  # Error: the last argument must be a code block.
                @static if $package == $PKG_CUDA
                    @test_throws ArgumentError hide_communication_cuda(:boundary_width, :(@parallel f()))                                                                      # Error: missing (bc and) communication code.
                    @test_throws ArgumentError hide_communication_cuda(:boundary_width, :(communication()))                                                                    # Error: missing @parallel call.
                    @test_throws ArgumentError hide_communication_cuda(:boundary_width, :(begin @parallel f(); end))                                                           # Error: missing (bc and) communication code.
                    @test_throws ArgumentError hide_communication_cuda(:boundary_width, :(begin communication(); end))                                                         # Error: missing @parallel call.
                    block = :(begin do_something(); @parallel f(); communication(); end);          @test_throws ArgumentError hide_communication_cuda(:boundary_width, block)  # Error: block does not start with @parallel call
                    block = :(begin @parallel stream=mystream f(); communication(); end);          @test_throws ArgumentError hide_communication_cuda(:boundary_width, block)  # Error: invalid keyword argument 'stream'.
                    block = :(begin @parallel ranges f(); communication(); end);                   @test_throws ArgumentError hide_communication_cuda(:boundary_width, block)  # Error: invalid optional argument 'ranges'.
                    block = :(begin @parallel nblocks nthreads f(); communication(); end);         @test_throws ArgumentError hide_communication_cuda(:boundary_width, block)  # Error: invalid optional arguments 'nblocks' and 'nthreads'.
                    block = :(begin @parallel ranges nblocks nthreads f(); communication(); end);  @test_throws ArgumentError hide_communication_cuda(:boundary_width, block)  # Error: invalid optional arguments 'ranges', 'nblocks' and 'nthreads'.

                    @test_throws ArgumentError hide_communication_cuda(:ranges_outer, :ranges_inner, :(@parallel f()))                                                                      # Error: missing (bc and) communication code.
                    @test_throws ArgumentError hide_communication_cuda(:ranges_outer, :ranges_inner, :(communication()))                                                                    # Error: missing @parallel call.
                    @test_throws ArgumentError hide_communication_cuda(:ranges_outer, :ranges_inner, :(begin @parallel f(); end))                                                           # Error: missing (bc and) communication code.
                    @test_throws ArgumentError hide_communication_cuda(:ranges_outer, :ranges_inner, :(begin communication(); end))                                                         # Error: missing @parallel call.
                    block = :(begin do_something(); @parallel f(); communication(); end);          @test_throws ArgumentError hide_communication_cuda(:ranges_outer, :ranges_inner, block)  # Error: block does not start with @parallel call
                    block = :(begin @parallel stream=mystream f(); communication(); end);          @test_throws ArgumentError hide_communication_cuda(:ranges_outer, :ranges_inner, block)  # Error: invalid keyword argument 'stream'.
                    block = :(begin @parallel ranges f(); communication(); end);                   @test_throws ArgumentError hide_communication_cuda(:ranges_outer, :ranges_inner, block)  # Error: invalid optional arguments 'ranges'.
                    block = :(begin @parallel nblocks nthreads f(); communication(); end);         @test_throws ArgumentError hide_communication_cuda(:ranges_outer, :ranges_inner, block)  # Error: invalid optional arguments 'nblocks' and 'nthreads'.
                    block = :(begin @parallel ranges nblocks nthreads f(); communication(); end);  @test_throws ArgumentError hide_communication_cuda(:ranges_outer, :ranges_inner, block)  # Error: invalid optional arguments 'ranges', 'nblocks' and 'nthreads'.

                    block = :(begin @parallel f(); @parallel g(); communication(); end);                          @test_throws ArgumentError hide_communication_cuda(:ranges_outer, :ranges_inner, block)  # Error: missing arguments 'ranges' in @parallel call for bc computations.
                    block = :(begin @parallel f(); @parallel stream=mystream g(); communication(); end);          @test_throws ArgumentError hide_communication_cuda(:ranges_outer, :ranges_inner, block)  # Error: invalid keyword argument in @parallel call for bc computations.
                    block = :(begin @parallel f(); @parallel nblocks nthreads g(); communication(); end);         @test_throws ArgumentError hide_communication_cuda(:ranges_outer, :ranges_inner, block)  # Error: invalid optional arguments 'nblocks' and 'nthreads' in @parallel call for bc computations.
                    block = :(begin @parallel f(); @parallel ranges nblocks nthreads g(); communication(); end);  @test_throws ArgumentError hide_communication_cuda(:ranges_outer, :ranges_inner, block)  # Error: invalid optional arguments 'ranges', 'nblocks' and 'nthreads' in @parallel call for bc computations.
                end
            end;
            @reset_parallel_kernel()
        end;
    end;
)) end == nothing || true;
