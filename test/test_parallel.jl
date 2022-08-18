using Test
using ParallelStencil
import ParallelStencil: @reset_parallel_stencil, @is_initialized, SUPPORTED_PACKAGES, PKG_CUDA, PKG_THREADS, INDICES
import ParallelStencil: @require, @prettystring, @gorgeousstring
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
            @require !@is_initialized()
            @init_parallel_stencil($package, Float64, 3)
            @require @is_initialized()
            @testset "@parallel kernelcall" begin # NOTE: calls must go to ParallelStencil.ParallelKernel.parallel and must therefore give the same result as in ParallelKernel (tests copied 1-to-1 from there).
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
            @testset "@parallel kernel" begin
                @testset "addition of range arguments" begin
                    expansion = @gorgeousstring(1, @parallel f(A, B, c::T) where T <: Integer = (@all(A) = @all(B)^c; return))
                    @test occursin("f(A, B, c::T, ranges::Tuple{UnitRange, UnitRange, UnitRange}, rangelength_x::Int64, rangelength_y::Int64, rangelength_z::Int64", expansion)
                end
                @testset "Data.Array to Data.DeviceArray" begin
                    @static if $package == $PKG_CUDA
                            expansion = @prettystring(1, @parallel f(A::Data.Array, B::Data.Array, c::T) where T <: Integer = (@all(A) = @all(B)^c; return))
                            @test occursin("f(A::Data.DeviceArray, B::Data.DeviceArray,", expansion)
                    end
                end
                @testset "Data.Cell to Data.DeviceCell" begin
                    @static if $package == $PKG_CUDA
                            expansion = @prettystring(1, @parallel f(A::Data.Cell, B::Data.Cell, c::T) where T <: Integer = (@all(A) = @all(B)^c; return))
                            @test occursin("f(A::Data.DeviceCell, B::Data.DeviceCell,", expansion)
                    end
                end
                @testset "Data.CellArray to Data.DeviceCellArray" begin
                    @static if $package == $PKG_CUDA
                            expansion = @prettystring(1, @parallel f(A::Data.CellArray, B::Data.CellArray, c::T) where T <: Integer = (@all(A) = @all(B)^c; return))
                            @test occursin("f(A::Data.DeviceCellArray, B::Data.DeviceCellArray,", expansion)
                    end
                end
                @testset "Data.TArray to Data.DeviceTArray" begin
                    @static if $package == $PKG_CUDA
                            expansion = @prettystring(1, @parallel f(A::Data.TArray, B::Data.TArray, c::T) where T <: Integer = (@all(A) = @all(B)^c; return))
                            @test occursin("f(A::Data.DeviceTArray, B::Data.DeviceTArray,", expansion)
                    end
                end
                @testset "Data.TCell to Data.DeviceTCell" begin
                    @static if $package == $PKG_CUDA
                            expansion = @prettystring(1, @parallel f(A::Data.TCell, B::Data.TCell, c::T) where T <: Integer = (@all(A) = @all(B)^c; return))
                            @test occursin("f(A::Data.DeviceTCell, B::Data.DeviceTCell,", expansion)
                    end
                end
                @testset "Data.TCellArray to Data.DeviceTCellArray" begin
                    @static if $package == $PKG_CUDA
                            expansion = @prettystring(1, @parallel f(A::Data.TCellArray, B::Data.TCellArray, c::T) where T <: Integer = (@all(A) = @all(B)^c; return))
                            @test occursin("f(A::Data.DeviceTCellArray, B::Data.DeviceTCellArray,", expansion)
                    end
                end
                @testset "@parallel kernel (3D)" begin
                    A  = @zeros(4, 5, 6)
                    @parallel function write_indices!(A)
                        @all(A) = $ix + ($iy-1)*size(A,1) + ($iz-1)*size(A,1)*size(A,2); # NOTE: $ix, $iy, $iz come from ParallelStencil.INDICES.
                        return
                    end
                    @parallel write_indices!(A);
                    @test all(Array(A) .== [ix + (iy-1)*size(A,1) + (iz-1)*size(A,1)*size(A,2) for ix=1:size(A,1), iy=1:size(A,2), iz=1:size(A,3)])
                end
            end;
            @testset "apply masks" begin
                expansion = @prettystring(1, @parallel sum!(A, B) = (@all(A) = @all(A) + @all(B); return))
                @test occursin("if @within(\"@all\", A)", expansion)
                @test @prettystring(@within("@all", A)) == string(:($ix <= size(A, 1) && ($iy <= size(A, 2) && $iz <= size(A, 3))))
            end;
            @reset_parallel_stencil()
        end;
        @testset "2. parallel macros (numbertype ommited)" begin
            @require !@is_initialized()
            @init_parallel_stencil(package = $package, ndims = 3)
            @require @is_initialized
            @testset "Data.Array{T} to Data.DeviceArray{T}" begin
                @static if $package == $PKG_CUDA
                    expansion = @prettystring(1, @parallel f(A::Data.Array{T}, B::Data.Array{T}, c<:Integer) where T <: PSNumber = (@all(A) = @all(B)^c; return))
                    @test occursin("f(A::Data.DeviceArray{T}, B::Data.DeviceArray{T},", expansion)
                end
            end
            @testset "Data.Cell{T} to Data.DeviceCell{T}" begin
                @static if $package == $PKG_CUDA
                    expansion = @prettystring(1, @parallel f(A::Data.Cell{T}, B::Data.Cell{T}, c<:Integer) where T <: PSNumber = (@all(A) = @all(B)^c; return))
                    @test occursin("f(A::Data.DeviceCell{T}, B::Data.DeviceCell{T},", expansion)
                end
            end
            @testset "Data.CellArray{T} to Data.DeviceCellArray{T}" begin
                @static if $package == $PKG_CUDA
                    expansion = @prettystring(1, @parallel f(A::Data.CellArray{T}, B::Data.CellArray{T}, c<:Integer) where T <: PSNumber = (@all(A) = @all(B)^c; return))
                    @test occursin("f(A::Data.DeviceCellArray{T}, B::Data.DeviceCellArray{T},", expansion)
                end
            end
            @reset_parallel_stencil()
        end
        @testset "3. Exceptions" begin
            @init_parallel_stencil($package, Float64, 3)
            @require @is_initialized
            @testset "arguments @parallel" begin
                @test_throws ArgumentError checkargs_parallel();                                                  # Error: isempty(args)
                @test_throws ArgumentError checkargs_parallel(:(f()), :(something));                              # Error: last arg is not function or a kernel call.
                @test_throws ArgumentError checkargs_parallel(:(f()=99), :(something));                           # Error: last arg is not function or a kernel call.
                @test_throws ArgumentError checkargs_parallel(:(f(;s=1)));                                        # Error: function call with keyword argument.
                @test_throws ArgumentError checkargs_parallel(:(f(;s=1)=(99*s; return)))                          # Error: function definition with keyword argument.
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
