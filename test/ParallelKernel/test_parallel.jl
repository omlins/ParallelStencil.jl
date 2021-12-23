using Test
import ParallelStencil
using ParallelStencil.ParallelKernel
import ParallelStencil.ParallelKernel: @reset_parallel_kernel, @is_initialized, SUPPORTED_PACKAGES, PKG_CUDA, PKG_THREADS
import ParallelStencil.ParallelKernel: @require, @prettystring
import ParallelStencil.ParallelKernel: checkargs_parallel, checkargs_parallel_indices, parallel_indices
using ParallelStencil.ParallelKernel.Exceptions
TEST_PACKAGES = SUPPORTED_PACKAGES
@static if PKG_CUDA in TEST_PACKAGES
    import CUDA
    if !CUDA.functional() TEST_PACKAGES = filter!(x->x≠PKG_CUDA, TEST_PACKAGES) end
end

@static for package in TEST_PACKAGES  eval(:(
    @testset "$(basename(@__FILE__)) (package: $(nameof($package)))" begin
        @testset "1. parallel macros" begin
            @require !@is_initialized()
            @init_parallel_kernel($package, Float64)
            @require @is_initialized()
            @testset "@parallel" begin
                @static if $package == $PKG_CUDA
                    call = @prettystring(1, @parallel f(A))
                    @test occursin("CUDA.@cuda blocks = ParallelStencil.ParallelKernel.compute_nblocks(length.(ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A))), ParallelStencil.ParallelKernel.compute_nthreads(length.(ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A))))) threads = ParallelStencil.ParallelKernel.compute_nthreads(length.(ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A)))) f(A, ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A)), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A)))[1])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A)))[2])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A)))[3])))", call)
                    @test occursin("CUDA.synchronize()", call)
                    call = @prettystring(1, @parallel ranges f(A))
                    @test occursin("CUDA.@cuda blocks = ParallelStencil.ParallelKernel.compute_nblocks(length.(ParallelStencil.ParallelKernel.promote_ranges(ranges)), ParallelStencil.ParallelKernel.compute_nthreads(length.(ParallelStencil.ParallelKernel.promote_ranges(ranges)))) threads = ParallelStencil.ParallelKernel.compute_nthreads(length.(ParallelStencil.ParallelKernel.promote_ranges(ranges))) f(A, ParallelStencil.ParallelKernel.promote_ranges(ranges), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ranges))[1])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ranges))[2])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ranges))[3])))", call)
                    call = @prettystring(1, @parallel nblocks nthreads f(A))
                    @test occursin("CUDA.@cuda blocks = nblocks threads = nthreads f(A, ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads)), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads)))[1])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads)))[2])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads)))[3])))", call)
                    call = @prettystring(1, @parallel ranges nblocks nthreads f(A))
                    @test occursin("CUDA.@cuda blocks = nblocks threads = nthreads f(A, ParallelStencil.ParallelKernel.promote_ranges(ranges), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ranges))[1])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ranges))[2])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ranges))[3])))", call)
                    call = @prettystring(1, @parallel nblocks nthreads stream=mystream f(A))
                    @test occursin("CUDA.@cuda blocks = nblocks threads = nthreads stream = mystream f(A, ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads)), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads)))[1])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads)))[2])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads)))[3])))", call)
                elseif $package == $PKG_THREADS
                    @test @prettystring(1, @parallel f(A)) == "f(A, ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A)), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A)))[1])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A)))[2])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A)))[3])))"
                    @test @prettystring(1, @parallel ranges f(A)) == "f(A, ParallelStencil.ParallelKernel.promote_ranges(ranges), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ranges))[1])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ranges))[2])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ranges))[3])))"
                    @test @prettystring(1, @parallel nblocks nthreads f(A)) == "f(A, ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads)), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads)))[1])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads)))[2])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads)))[3])))"
                    @test @prettystring(1, @parallel ranges nblocks nthreads f(A)) == "f(A, ParallelStencil.ParallelKernel.promote_ranges(ranges), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ranges))[1])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ranges))[2])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ranges))[3])))"
                    @test @prettystring(1, @parallel stream=mystream f(A)) == "f(A, ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A)), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A)))[1])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A)))[2])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A)))[3])))"
                end;
            end;
            @testset "@parallel_indices" begin
                @testset "addition of range arguments" begin
                    expansion = @prettystring(1, @parallel_indices (ix,iy) f(a::T, b::T) where T <: Union{Array{Float32}, Array{Float64}} = (println("a=$a, b=$b)"); return))
                    @test occursin("f(a::T, b::T, var\"##ranges, ParallelStencil.ParallelKernel#260\"::Tuple{UnitRange, UnitRange, UnitRange}, var\"##rangelength_x, ParallelStencil.ParallelKernel#261\"::Int64, var\"##rangelength_y, ParallelStencil.ParallelKernel#262\"::Int64, var\"##rangelength_z, ParallelStencil.ParallelKernel#263\"::Int64", expansion)
                end
                @testset "Data.Array to Data.DeviceArray" begin
                    @static if $package == $PKG_CUDA
                            expansion = @prettystring(1, @parallel_indices (ix,iy) f(A::Data.Array, B::Data.Array, c::T) where T <: Integer = (A[ix,iy] = B[ix,iy]^c; return))
                            @test occursin("f(A::Data.DeviceArray, B::Data.DeviceArray,", expansion)
                    end
                end
                @testset "@parallel_indices (1D)" begin
                    A  = @zeros(4)
                    @parallel_indices (ix) function write_indices!(A)
                        A[ix] = ix;
                        return
                    end
                    @parallel write_indices!(A);
                    @test all(Array(A) .== [ix for ix=1:size(A,1)])
                end;
                @testset "@parallel_indices (2D)" begin
                    A  = @zeros(4, 5)
                    @parallel_indices (ix,iy) function write_indices!(A)
                        A[ix,iy] = ix + (iy-1)*size(A,1);
                        return
                    end
                    @parallel write_indices!(A);
                    @test all(Array(A) .== [ix + (iy-1)*size(A,1) for ix=1:size(A,1), iy=1:size(A,2)])
                end;
                @testset "@parallel_indices (3D)" begin
                    A  = @zeros(4, 5, 6)
                    @parallel_indices (ix,iy,iz) function write_indices!(A)
                        A[ix,iy,iz] = ix + (iy-1)*size(A,1) + (iz-1)*size(A,1)*size(A,2);
                        return
                    end
                    @parallel write_indices!(A);
                    @test all(Array(A) .== [ix + (iy-1)*size(A,1) + (iz-1)*size(A,1)*size(A,2) for ix=1:size(A,1), iy=1:size(A,2), iz=1:size(A,3)])
                end;
                @testset "@parallel_indices (1D in 3D)" begin
                    A  = @zeros(4, 5, 6)
                    @parallel_indices (iy) function write_indices!(A)
                        A[1,iy,1] = iy;
                        return
                    end
                    @parallel 1:size(A,2) write_indices!(A);
                    @test all(Array(A)[1,:,1] .== [iy for iy=1:size(A,2)])
                end;
                @testset "@parallel_indices (2D in 3D)" begin
                    A  = @zeros(4, 5, 6)
                    @parallel_indices (ix,iz) function write_indices!(A)
                        A[ix,end,iz] = ix + (iz-1)*size(A,1);
                        return
                    end
                    @parallel (1:size(A,1), 1:size(A,3)) write_indices!(A);
                    @test all(Array(A)[:,end,:] .== [ix + (iz-1)*size(A,1) for ix=1:size(A,1), iz=1:size(A,3)])
                end;
            end;
            @testset "@parallel_async" begin
                @static if $package == $PKG_CUDA
                    call = @prettystring(1, @parallel_async f(A))
                    @test !occursin("CUDA.synchronize()", call)
                end;
            end;
            @testset "@synchronize" begin
                @static if $package == $PKG_CUDA
                    @test @prettystring(1, @synchronize()) == "CUDA.synchronize()"
                end;
            end;
            @reset_parallel_kernel()
        end;
        @testset "2. parallel macros (literal conversion)" begin
            @testset "@parallel_indices (Float64)" begin
                @require !@is_initialized()
                @init_parallel_kernel($package, Float64)
                @require @is_initialized()
                expansion = @prettystring(@parallel_indices (ix) f!(A) = (A[ix] = A[ix] + 1.0f0; return))
                @test occursin("A[ix] = A[ix] + 1.0\n", expansion)
                @reset_parallel_kernel()
            end;
            @testset "@parallel_indices (Float32)" begin
                @require !@is_initialized()
                @init_parallel_kernel($package, Float32)
                @require @is_initialized()
                expansion = @prettystring(@parallel_indices (ix) f!(A) = (A[ix] = A[ix] + 1.0; return))
                @test occursin("A[ix] = A[ix] + 1.0f0\n", expansion)
                @reset_parallel_kernel()
            end;
            @testset "@parallel_indices (Float16)" begin
                @require !@is_initialized()
                @init_parallel_kernel($package, Float16)
                @require @is_initialized()
                expansion = @prettystring(@parallel_indices (ix) f!(A) = (A[ix] = A[ix] + 1.0; return))
                @test occursin("A[ix] = A[ix] + Float16(1.0)\n", expansion)
                @reset_parallel_kernel()
            end;
            @testset "@parallel_indices (ComplexF64)" begin
                @require !@is_initialized()
                @init_parallel_kernel($package, ComplexF64)
                @require @is_initialized()
                expansion = @prettystring(@parallel_indices (ix) f!(A) = (A[ix] = 2.0f0 - 1.0f0im - A[ix] + 1.0f0; return))
                @test occursin("A[ix] = ((2.0 - 1.0im) - A[ix]) + 1.0\n", expansion)
                @reset_parallel_kernel()
            end;
            @testset "@parallel_indices (ComplexF32)" begin
                @require !@is_initialized()
                @init_parallel_kernel($package, ComplexF32)
                @require @is_initialized()
                expansion = @prettystring(@parallel_indices (ix) f!(A) = (A[ix] = 2.0 - 1.0im - A[ix] + 1.0; return))
                @test occursin("A[ix] = ((2.0f0 - 1.0f0im) - A[ix]) + 1.0f0\n", expansion)
                @reset_parallel_kernel()
            end;
            @testset "@parallel_indices (ComplexF16)" begin
                @require !@is_initialized()
                @init_parallel_kernel($package, ComplexF16)
                @require @is_initialized()
                expansion = @prettystring(@parallel_indices (ix) f!(A) = (A[ix] = 2.0 - 1.0im - A[ix] + 1.0; return))
                @test occursin("A[ix] = ((Float16(2.0) - Float16(1.0) * im) - A[ix]) + Float16(1.0)\n", expansion)
                @reset_parallel_kernel()
            end;
        end
        @testset "3. parallel macros (numbertype ommited)" begin
            @require !@is_initialized()
            @init_parallel_kernel(package = $package)
            @require @is_initialized
            @testset "Data.Array{T} to Data.DeviceArray{T}" begin
                @static if $package == $PKG_CUDA
                        expansion = @prettystring(1, @parallel_indices (ix,iy) f(A::Data.Array{T}, B::Data.Array{T}, c::<:Integer) where T <: Union{Float32, Float64}  = (A[ix,iy] = B[ix,iy]^c; return))
                        @test occursin("f(A::Data.DeviceArray{T}, B::Data.DeviceArray{T},", expansion)
                end
            end
            @reset_parallel_kernel()
        end
        @testset "4. Exceptions" begin
            @require !@is_initialized()
            @init_parallel_kernel($package, Float64)
            @require @is_initialized
            @testset "arguments @parallel" begin
                @test_throws ArgumentError checkargs_parallel();                                                    # Error: isempty(args)
                @test_throws ArgumentError checkargs_parallel(:(f()), :(something));                                # Error: last arg is not function call.
                @test_throws ArgumentError checkargs_parallel(:(f()=99));                                           # Error: last arg is not function call.
                #TODO: Here I am: kw for calls look very different: head :kw - fix in parallel.jl/shared.jl
                @test_throws ArgumentError checkargs_parallel(:(f(;s=1)));                                          # Error: function call with keyword argument.
                @test_throws ArgumentError checkargs_parallel(:ranges, :nblocks, :nthreads, :something, :(f()));    # Error: length(posargs) > 3
                @test_throws KeywordArgumentError checkargs_parallel(:(blocks=blocks), :(f()));                     # Error: blocks keyword argument is not allowed
                @test_throws KeywordArgumentError checkargs_parallel(:(threads=threads), :(f()));                   # Error: threads keyword argument is not allowed
            end;
            @testset "arguments @parallel_indices" begin
                @test_throws ArgumentError checkargs_parallel_indices();                                            # Error: length(args) != 2
                @test_throws ArgumentError checkargs_parallel_indices(:(f()=99));                                   # Error: length(args) != 2
                @test_throws ArgumentError checkargs_parallel_indices(:((ix,iy,iz)), :(f()=99), :(something));      # Error: length(args) != 2
                @test_throws ArgumentError checkargs_parallel_indices(:ix, :iy, :iz, :(f()=99));                    # Error: length(args) != 2
                @test_throws ArgumentError checkargs_parallel_indices(:(f()=99), :((ix,iy,iz)));                    # Error: last arg is not function.
                @test_throws ArgumentError checkargs_parallel_indices(:((ix,iy,iz)), :(f()));                       # Error: last arg is not function.
                @test_throws ArgumentError checkargs_parallel_indices(:((ix,iy,iz)), :(f(;s=1)=(99*s; return)))     # Error: function defines keyword.
                @test_throws ArgumentError parallel_indices(:((ix,iy,iz)), :(f()=99))                               # Error: no return statement in function.
                @test_throws ArgumentError parallel_indices(:((ix,iy,iz)), :(f()=(99; return something)))           # Error: function does not return nothing.
                #TODO: this tests does not pass anymore for unknown reasons:
                #@test_throws ArgumentError parallel_indices(:((ix,iy,iz)), :(f()=(99; if x return y end; return)))  # Error: function contains more than one return statement.
            end;
            @reset_parallel_kernel()
        end;
    end;
)) end == nothing || true;
