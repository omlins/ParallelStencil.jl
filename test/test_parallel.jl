using Test
using ParallelStencil
import ParallelStencil: @reset_parallel_stencil, @is_initialized, SUPPORTED_PACKAGES, PKG_CUDA, PKG_THREADS, INDICES
import ParallelStencil: @require, longnameof, @prettyexpand, prettystring
import ParallelStencil: checkargs_parallel, validate_body, parallel
using ParallelStencil.Exceptions
using ParallelStencil.FiniteDifferences3D
ix, iy, iz = INDICES[1], INDICES[2], INDICES[3]
TEST_PACKAGES = SUPPORTED_PACKAGES
@static if PKG_CUDA in TEST_PACKAGES
    import CUDA
    if !CUDA.functional() TEST_PACKAGES = filter!(x->x≠PKG_CUDA, TEST_PACKAGES) end
end

@static for package in TEST_PACKAGES  eval(:(
    @testset "$(basename(@__FILE__)) (package: $(nameof($package)))" begin
        @testset "1. parallel macros" begin
            @init_parallel_stencil($package, Float64, 3)
            @require @is_initialized()
            @testset "@parallel kernelcall" begin # NOTE: calls must go to ParallelStencil.ParallelKernel.parallel and must therefore give the same result as in ParallelKernel (tests copied from there).
                @static if $package == PKG_CUDA
                    call = prettystring(parallel(:(f(A))))
                    @test occursin("CUDA.@cuda blocks = ParallelStencil.ParallelKernel.compute_nblocks(length.(ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A))), ParallelStencil.ParallelKernel.compute_nthreads(length.(ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A))))) threads = ParallelStencil.ParallelKernel.compute_nthreads(length.(ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A)))) f(A, ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A)))", call)
                    @test occursin("CUDA.synchronize()", call)
                    call = prettystring(parallel(:ranges, :(f(A))))
                    @test occursin("CUDA.@cuda blocks = ParallelStencil.ParallelKernel.compute_nblocks(length.(ParallelStencil.ParallelKernel.promote_ranges(ranges)), ParallelStencil.ParallelKernel.compute_nthreads(length.(ParallelStencil.ParallelKernel.promote_ranges(ranges)))) threads = ParallelStencil.ParallelKernel.compute_nthreads(length.(ParallelStencil.ParallelKernel.promote_ranges(ranges))) f(A, ParallelStencil.ParallelKernel.promote_ranges(ranges))", call)
                    call = prettystring(parallel(:nblocks, :nthreads, :(f(A))))
                    @test occursin("CUDA.@cuda blocks = nblocks threads = nthreads f(A, ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads)))", call)
                    call = prettystring(parallel(:ranges, :nblocks, :nthreads, :(f(A))))
                    @test occursin("CUDA.@cuda blocks = nblocks threads = nthreads f(A, ParallelStencil.ParallelKernel.promote_ranges(ranges))", call)
                    call = prettystring(parallel(:nblocks, :nthreads, :(stream=mystream), :(f(A))))
                    @test occursin("CUDA.@cuda blocks = nblocks threads = nthreads stream = mystream f(A, ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads)))", call)
                elseif $package == PKG_THREADS
                    @test prettystring(parallel(:(f(A)))) == "f(A, ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A)))"
                    @test prettystring(parallel(:ranges, :(f(A)))) == "f(A, ParallelStencil.ParallelKernel.promote_ranges(ranges))"
                    @test prettystring(parallel(:nblocks, :nthreads, :(f(A)))) == "f(A, ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads)))"
                    @test prettystring(parallel(:ranges, :nblocks, :nthreads, :(f(A)))) == "f(A, ParallelStencil.ParallelKernel.promote_ranges(ranges))"
                    @test prettystring(parallel(:(stream=mystream), :(f(A)))) == "f(A, ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A)))"
                end;
            end;
            @testset "@parallel kernel" begin
                @require @is_initialized()
                A  = @zeros(4, 5, 6)
                @parallel function write_indices!(A)
                    @all(A) = $ix + ($iy-1)*size(A,1) + ($iz-1)*size(A,1)*size(A,2); # NOTE: $ix, $iy, $iz come from ParallelStencil.INDICES.
                    return
                end
                @parallel write_indices!(A);
                @test all(Array(A) .== [ix + (iy-1)*size(A,1) + (iz-1)*size(A,1)*size(A,2) for ix=1:size(A,1), iy=1:size(A,2), iz=1:size(A,3)])
            end;
            @testset "apply masks" begin
                @require @is_initialized()
                expansion = string(@prettyexpand 1 @parallel sum!(A, B) = (@all(A) = @all(A) + @all(B); return))
                @test occursin("if @within(\"@all\", A)", expansion)
                @test string(@prettyexpand @within("@all", A)) == string(:($ix <= size(A, 1) && ($iy <= size(A, 2) && $iz <= size(A, 3))))
            end;
            @reset_parallel_stencil()
        end;
        @testset "2. Exceptions" begin
            @init_parallel_stencil($package, Float64, 3)
            @require @is_initialized
            @testset "arguments @parallel" begin
                @test_throws ArgumentError checkargs_parallel();                                                  # Error: isempty(args)
                @test_throws ArgumentError checkargs_parallel(:(f()), :(something));                              # Error: last arg is not function or a kernel call.
                @test_throws ArgumentError checkargs_parallel(:(f()=99), :(something));                           # Error: last arg is not function or a kernel call.
                @test_throws ArgumentError checkargs_parallel(:(something), :(f()=99));                           # Error: if last arg is a function, then there cannot be any other argument.
                @test_throws ArgumentError checkargs_parallel(:ranges, :nblocks, :nthreads, :something, :(f()));  # Error: if last arg is a kernel call, then we have an error if length(posargs) > 3
                @test_throws ArgumentError validate_body(:(a = b + 1))
                @test_throws ArgumentError validate_body(:(a = b + 1; @all(A) = @all(B) + 1))
                @test_throws ArgumentError validate_body(:(A = @all(B) + 1; @all(A) = @all(B) + 1))
            end;
            @reset_parallel_stencil()
        end;
    end;
)) end == nothing || true;
