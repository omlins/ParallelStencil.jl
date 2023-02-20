using Test
using ParallelStencil
import ParallelStencil: @reset_parallel_stencil, @is_initialized, SUPPORTED_PACKAGES, PKG_CUDA, PKG_AMDGPU, PKG_THREADS, INDICES
import ParallelStencil: @require, @prettystring, @gorgeousstring, @isgpu
import ParallelStencil: checkargs_parallel, validate_body, parallel
using ParallelStencil.Exceptions
using ParallelStencil.FiniteDifferences3D
ix, iy, iz = INDICES[1], INDICES[2], INDICES[3]
TEST_PACKAGES = SUPPORTED_PACKAGES
@static if PKG_CUDA in TEST_PACKAGES
    import CUDA
    if !CUDA.functional() TEST_PACKAGES = filter!(x->x≠PKG_CUDA, TEST_PACKAGES) end
end
@static if PKG_AMDGPU in TEST_PACKAGES
    import AMDGPU
    if !AMDGPU.functional() TEST_PACKAGES = filter!(x->x≠PKG_AMDGPU, TEST_PACKAGES) end
end

import ParallelStencil.@gorgeousexpand

@static for package in TEST_PACKAGES  eval(:(
    @testset "$(basename(@__FILE__)) (package: $(nameof($package)))" begin
        @testset "1. parallel macros" begin
            @require !@is_initialized()
            @init_parallel_stencil($package, Float64, 3)
            @require @is_initialized()
            @testset "@parallel <kernelcall>" begin # NOTE: calls must go to ParallelStencil.ParallelKernel.parallel and must therefore give the same result as in ParallelKernel (tests copied 1-to-1 from there).
                @static if $package == $PKG_CUDA
                    call = @prettystring(1, @parallel f(A))
                    @test occursin("CUDA.@cuda blocks = ParallelStencil.ParallelKernel.compute_nblocks(length.(ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A))), ParallelStencil.ParallelKernel.compute_nthreads(length.(ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A))))) threads = ParallelStencil.ParallelKernel.compute_nthreads(length.(ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A)))) stream = CUDA.stream() f(A, ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A)), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A)))[1])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A)))[2])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A)))[3])))", call)
                    @test occursin("CUDA.synchronize(CUDA.stream())", call)
                    call = @prettystring(1, @parallel ranges f(A))
                    @test occursin("CUDA.@cuda blocks = ParallelStencil.ParallelKernel.compute_nblocks(length.(ParallelStencil.ParallelKernel.promote_ranges(ranges)), ParallelStencil.ParallelKernel.compute_nthreads(length.(ParallelStencil.ParallelKernel.promote_ranges(ranges)))) threads = ParallelStencil.ParallelKernel.compute_nthreads(length.(ParallelStencil.ParallelKernel.promote_ranges(ranges))) stream = CUDA.stream() f(A, ParallelStencil.ParallelKernel.promote_ranges(ranges), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ranges))[1])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ranges))[2])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ranges))[3])))", call)
                    call = @prettystring(1, @parallel nblocks nthreads f(A))
                    @test occursin("CUDA.@cuda blocks = nblocks threads = nthreads stream = CUDA.stream() f(A, ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads)), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads)))[1])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads)))[2])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads)))[3])))", call)
                    call = @prettystring(1, @parallel ranges nblocks nthreads f(A))
                    @test occursin("CUDA.@cuda blocks = nblocks threads = nthreads stream = CUDA.stream() f(A, ParallelStencil.ParallelKernel.promote_ranges(ranges), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ranges))[1])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ranges))[2])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ranges))[3])))", call)
                    call = @prettystring(1, @parallel nblocks nthreads stream=mystream f(A))
                    @test occursin("CUDA.@cuda blocks = nblocks threads = nthreads stream = mystream f(A, ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads)), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads)))[1])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads)))[2])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads)))[3])))", call)
                    call = @prettystring(2, @parallel loopopt=true f(A))
                    @test occursin("CUDA.@cuda blocks = ParallelStencil.ParallelKernel.compute_nblocks(cld.(length.(ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A))), (1, 1, 16)), ParallelStencil.compute_nthreads_loopopt(cld.(length.(ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A))), (1, 1, 16)), 3, (-1:1, -1:1, -1:1))) threads = ParallelStencil.compute_nthreads_loopopt(cld.(length.(ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A))), (1, 1, 16)), 3, (-1:1, -1:1, -1:1)) stream = CUDA.stream() shmem = ((ParallelStencil.compute_nthreads_loopopt(cld.(length.(ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A))), (1, 1, 16)), 3, (-1:1, -1:1, -1:1)))[1] + 3) * ((ParallelStencil.compute_nthreads_loopopt(cld.(length.(ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A))), (1, 1, 16)), 3, (-1:1, -1:1, -1:1)))[2] + 3) * sizeof(Float64) f(A, ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A)), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A)))[1])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A)))[2])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A)))[3])))", call)
                    call = @prettystring(2, @parallel ranges loopopt=true f(A))
                    @test occursin("CUDA.@cuda blocks = ParallelStencil.ParallelKernel.compute_nblocks(cld.(length.(ParallelStencil.ParallelKernel.promote_ranges(ranges)), (1, 1, 16)), ParallelStencil.compute_nthreads_loopopt(cld.(length.(ParallelStencil.ParallelKernel.promote_ranges(ranges)), (1, 1, 16)), 3, (-1:1, -1:1, -1:1))) threads = ParallelStencil.compute_nthreads_loopopt(cld.(length.(ParallelStencil.ParallelKernel.promote_ranges(ranges)), (1, 1, 16)), 3, (-1:1, -1:1, -1:1)) stream = CUDA.stream() shmem = ((ParallelStencil.compute_nthreads_loopopt(cld.(length.(ParallelStencil.ParallelKernel.promote_ranges(ranges)), (1, 1, 16)), 3, (-1:1, -1:1, -1:1)))[1] + 3) * ((ParallelStencil.compute_nthreads_loopopt(cld.(length.(ParallelStencil.ParallelKernel.promote_ranges(ranges)), (1, 1, 16)), 3, (-1:1, -1:1, -1:1)))[2] + 3) * sizeof(Float64) f(A, ParallelStencil.ParallelKernel.promote_ranges(ranges), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ranges))[1])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ranges))[2])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ranges))[3])))", call)
                elseif $package == $PKG_AMDGPU
                    call = @prettystring(1, @parallel f(A))
                    @test occursin("ParallelStencil.ParallelKernel.push_signal!(ParallelStencil.ParallelKernel.get_default_rocstream(), AMDGPU.@roc(gridsize = ParallelStencil.ParallelKernel.compute_nblocks(length.(ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A))), ParallelStencil.ParallelKernel.compute_nthreads(length.(ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A))))) .* ParallelStencil.ParallelKernel.compute_nthreads(length.(ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A)))), groupsize = ParallelStencil.ParallelKernel.compute_nthreads(length.(ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A)))), queue = (ParallelStencil.ParallelKernel.get_default_rocstream()).queue, f(A, ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A)), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A)))[1])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A)))[2])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A)))[3])))))", call)
                    @test occursin("ParallelStencil.ParallelKernel.synchronize_rocstream(ParallelStencil.ParallelKernel.get_default_rocstream())", call)
                    call = @prettystring(1, @parallel ranges f(A))
                    @test occursin("ParallelStencil.ParallelKernel.push_signal!(ParallelStencil.ParallelKernel.get_default_rocstream(), AMDGPU.@roc(gridsize = ParallelStencil.ParallelKernel.compute_nblocks(length.(ParallelStencil.ParallelKernel.promote_ranges(ranges)), ParallelStencil.ParallelKernel.compute_nthreads(length.(ParallelStencil.ParallelKernel.promote_ranges(ranges)))) .* ParallelStencil.ParallelKernel.compute_nthreads(length.(ParallelStencil.ParallelKernel.promote_ranges(ranges))), groupsize = ParallelStencil.ParallelKernel.compute_nthreads(length.(ParallelStencil.ParallelKernel.promote_ranges(ranges))), queue = (ParallelStencil.ParallelKernel.get_default_rocstream()).queue, f(A, ParallelStencil.ParallelKernel.promote_ranges(ranges), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ranges))[1])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ranges))[2])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ranges))[3])))))", call)
                    call = @prettystring(1, @parallel nblocks nthreads f(A))
                    @test occursin("ParallelStencil.ParallelKernel.push_signal!(ParallelStencil.ParallelKernel.get_default_rocstream(), AMDGPU.@roc(gridsize = nblocks .* nthreads, groupsize = nthreads, queue = (ParallelStencil.ParallelKernel.get_default_rocstream()).queue, f(A, ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads)), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads)))[1])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads)))[2])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads)))[3])))))", call)
                    call = @prettystring(1, @parallel ranges nblocks nthreads f(A))
                    @test occursin("ParallelStencil.ParallelKernel.push_signal!(ParallelStencil.ParallelKernel.get_default_rocstream(), AMDGPU.@roc(gridsize = nblocks .* nthreads, groupsize = nthreads, queue = (ParallelStencil.ParallelKernel.get_default_rocstream()).queue, f(A, ParallelStencil.ParallelKernel.promote_ranges(ranges), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ranges))[1])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ranges))[2])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ranges))[3])))))", call)
                    call = @prettystring(1, @parallel nblocks nthreads stream=mystream f(A))
                    @test occursin("ParallelStencil.ParallelKernel.push_signal!(mystream, AMDGPU.@roc(gridsize = nblocks .* nthreads, groupsize = nthreads, queue = mystream.queue, f(A, ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads)), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads)))[1])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads)))[2])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads)))[3])))))", call)
                elseif $package == $PKG_THREADS
                    @test @prettystring(1, @parallel f(A)) == "f(A, ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A)), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A)))[1])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A)))[2])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A)))[3])))"
                    @test @prettystring(1, @parallel ranges f(A)) == "f(A, ParallelStencil.ParallelKernel.promote_ranges(ranges), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ranges))[1])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ranges))[2])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ranges))[3])))"
                    @test @prettystring(1, @parallel nblocks nthreads f(A)) == "f(A, ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads)), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads)))[1])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads)))[2])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads)))[3])))"
                    @test @prettystring(1, @parallel ranges nblocks nthreads f(A)) == "f(A, ParallelStencil.ParallelKernel.promote_ranges(ranges), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ranges))[1])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ranges))[2])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ranges))[3])))"
                    @test @prettystring(1, @parallel stream=mystream f(A)) == "f(A, ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A)), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A)))[1])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A)))[2])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A)))[3])))"
                    @test @prettystring(2, @parallel loopopt=true f(A)) == "f(A, ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A)), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A)))[1])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A)))[2])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A)))[3])))"
                    @test @prettystring(2, @parallel ranges loopopt=true f(A)) == "f(A, ParallelStencil.ParallelKernel.promote_ranges(ranges), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ranges))[1])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ranges))[2])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ranges))[3])))"
                end;
            end;
            @testset "@parallel <kernel>" begin
                @testset "addition of range arguments" begin
                    expansion = @gorgeousstring(1, @parallel f(A, B, c::T) where T <: Integer = (@all(A) = @all(B)^c; return))
                    @test occursin("f(A, B, c::T, ranges::Tuple{UnitRange, UnitRange, UnitRange}, rangelength_x::Int64, rangelength_y::Int64, rangelength_z::Int64", expansion)
                end
                @testset "Data.Array to Data.DeviceArray" begin
                    @static if @isgpu($package)
                            expansion = @prettystring(1, @parallel f(A::Data.Array, B::Data.Array, c::T) where T <: Integer = (@all(A) = @all(B)^c; return))
                            @test occursin("f(A::Data.DeviceArray, B::Data.DeviceArray,", expansion)
                    end
                end
                @testset "Data.Cell to Data.DeviceCell" begin
                    @static if @isgpu($package)
                            expansion = @prettystring(1, @parallel f(A::Data.Cell, B::Data.Cell, c::T) where T <: Integer = (@all(A) = @all(B)^c; return))
                            @test occursin("f(A::Data.DeviceCell, B::Data.DeviceCell,", expansion)
                    end
                end
                @testset "Data.CellArray to Data.DeviceCellArray" begin
                    @static if @isgpu($package)
                            expansion = @prettystring(1, @parallel f(A::Data.CellArray, B::Data.CellArray, c::T) where T <: Integer = (@all(A) = @all(B)^c; return))
                            @test occursin("f(A::Data.DeviceCellArray, B::Data.DeviceCellArray,", expansion)
                    end
                end
                @testset "Data.TArray to Data.DeviceTArray" begin
                    @static if @isgpu($package)
                            expansion = @prettystring(1, @parallel f(A::Data.TArray, B::Data.TArray, c::T) where T <: Integer = (@all(A) = @all(B)^c; return))
                            @test occursin("f(A::Data.DeviceTArray, B::Data.DeviceTArray,", expansion)
                    end
                end
                @testset "Data.TCell to Data.DeviceTCell" begin
                    @static if @isgpu($package)
                            expansion = @prettystring(1, @parallel f(A::Data.TCell, B::Data.TCell, c::T) where T <: Integer = (@all(A) = @all(B)^c; return))
                            @test occursin("f(A::Data.DeviceTCell, B::Data.DeviceTCell,", expansion)
                    end
                end
                @testset "Data.TCellArray to Data.DeviceTCellArray" begin
                    @static if @isgpu($package)
                            expansion = @prettystring(1, @parallel f(A::Data.TCellArray, B::Data.TCellArray, c::T) where T <: Integer = (@all(A) = @all(B)^c; return))
                            @test occursin("f(A::Data.DeviceTCellArray, B::Data.DeviceTCellArray,", expansion)
                    end
                end
                @testset "@parallel <kernel> (3D)" begin
                    A  = @zeros(4, 5, 6)
                    @parallel function write_indices!(A)
                        @all(A) = $ix + ($iy-1)*size(A,1) + ($iz-1)*size(A,1)*size(A,2); # NOTE: $ix, $iy, $iz come from ParallelStencil.INDICES.
                        return
                    end
                    @parallel write_indices!(A);
                    @test all(Array(A) .== [ix + (iy-1)*size(A,1) + (iz-1)*size(A,1)*size(A,2) for ix=1:size(A,1), iy=1:size(A,2), iz=1:size(A,3)])
                end
                # TODO: activate these tests also for AMDGPU once shared memory is implemented there.
                @static if $package == $PKG_CUDA
                    @testset "@parallel loopopt <kernel>" begin
                        nx, ny, nz   = 32, 8, 8 # 6, 6, 6
                        # threads      = (8, 4, 1)
                        # blocks       = ceil.(Int, (nx/threads[1], ny/threads[2], nz/LOOPSIZE))
                        # shmem        = (threads[1]+2)*(threads[2]+2)*sizeof(Float64)
                        @testset "@parallel_indices <kernel> (3D, loopopt, stencilranges=0:0)" begin
                            A  = @zeros(nx, ny, nz);
                            A2 = @zeros(nx, ny, nz);
                            copy!(A, [ix + (iy-1)*size(A,1) + (iz-1)*size(A,1)*size(A,2) for ix=1:size(A,1), iy=1:size(A,2), iz=1:size(A,3)].^3);
                            @parallel_indices (ix,iy,iz) loopopt=true loopsize=3 function copy_loopopt!(A2, A)
                                A2[ix,iy,iz] = A[ix,iy,iz]
                                return
                            end
                            @parallel loopopt=true copy_loopopt!(A2, A);
                            @test all(Array(A2) .== Array(A))
                        end
                        @testset "@parallel <kernel> (3D, loopopt, stencilranges=0:0)" begin
                            A  = @zeros(nx, ny, nz);
                            A2 = @zeros(nx, ny, nz);
                            copy!(A, [ix + (iy-1)*size(A,1) + (iz-1)*size(A,1)*size(A,2) for ix=1:size(A,1), iy=1:size(A,2), iz=1:size(A,3)].^3);
                            @parallel loopopt=true loopsize=3 function copy_loopopt!(A2, A)
                                @all(A2) = @all(A)
                                return
                            end
                            @parallel loopopt=true copy_loopopt!(A2, A);
                            @test all(Array(A2) .== Array(A))
                        end
                        # Add here also single difference, where with parallel we have a negative halo: stencilranges=(0:0, 0:0, 1:2)
                        @testset "@parallel_indices <kernel> (3D, loopopt, stencilranges=(0:0, 0:0, -1:1); z-stencil)" begin
                            A      = @zeros(nx, ny, nz);
                            A2     = @zeros(nx, ny, nz);
                            A2_ref = @zeros(nx, ny, nz);
                            copy!(A, [ix + (iy-1)*size(A,1) + (iz-1)*size(A,1)*size(A,2) for ix=1:size(A,1), iy=1:size(A,2), iz=1:size(A,3)].^3);
                            @parallel_indices (ix,iy,iz) loopopt=true loopsize=3 function d2_loopopt!(A2, A)
                                if (iz>1 && iz<size(A2,3))
                                    A2[ix,iy,iz] = A[ix,iy,iz+1] - 2.0*A[ix,iy,iz] + A[ix,iy,iz-1]
                                end
                                return
                            end
                            @parallel loopopt=true d2_loopopt!(A2, A);
                            A2_ref[:,:,2:end-1] .= A[:,:,3:end] .- 2.0.*A[:,:,2:end-1] .+ A[:,:,1:end-2];
                            @test all(Array(A2) .== Array(A2_ref))
                        end
                        @testset "@parallel_indices <kernel> (3D, loopopt, stencilranges=(0:0, -1:1, 0:0); y-stencil)" begin
                            A      = @zeros(nx, ny, nz);
                            A2     = @zeros(nx, ny, nz);
                            A2_ref = @zeros(nx, ny, nz);
                            copy!(A, [ix + (iy-1)*size(A,1) + (iz-1)*size(A,1)*size(A,2) for ix=1:size(A,1), iy=1:size(A,2), iz=1:size(A,3)].^3);
                            @parallel_indices (ix,iy,iz) loopopt=true function d2_loopopt!(A2, A)
                                if (iy>1 && iy<size(A2,2))
                                    A2[ix,iy,iz] = A[ix,iy+1,iz] - 2.0*A[ix,iy,iz] + A[ix,iy-1,iz]
                                end
                                return
                            end
                            @parallel loopopt=true d2_loopopt!(A2, A);
                            A2_ref[:,2:end-1,:] .= A[:,3:end,:] .- 2.0.*A[:,2:end-1,:] .+ A[:,1:end-2,:];
                            @test all(Array(A2) .== Array(A2_ref))
                        end
                        @testset "@parallel <kernel> (3D, loopopt, stencilranges=(1:1, 1:1, 0:2); z-stencil)" begin
                            A      = @zeros(nx, ny, nz);
                            A2     = @zeros(nx, ny, nz);
                            A2_ref = @zeros(nx, ny, nz);
                            copy!(A, [ix + (iy-1)*size(A,1) + (iz-1)*size(A,1)*size(A,2) for ix=1:size(A,1), iy=1:size(A,2), iz=1:size(A,3)].^3);
                            @parallel loopopt=true loopsize=3 function d2_loopopt!(A2, A)
                                @inn(A2) = @d2_zi(A)
                                return
                            end
                            @parallel loopopt=true d2_loopopt!(A2, A);
                            A2_ref[2:end-1,2:end-1,2:end-1] .= A[2:end-1,2:end-1,3:end] .- 2.0.*A[2:end-1,2:end-1,2:end-1] .+ A[2:end-1,2:end-1,1:end-2];
                            @test all(Array(A2) .== Array(A2_ref))
                        end
                        @testset "@parallel <kernel> (3D, loopopt, stencilranges=(1:1, 0:2, 1:1); y-stencil)" begin
                            A      = @zeros(nx, ny, nz);
                            A2     = @zeros(nx, ny, nz);
                            A2_ref = @zeros(nx, ny, nz);
                            copy!(A, [ix + (iy-1)*size(A,1) + (iz-1)*size(A,1)*size(A,2) for ix=1:size(A,1), iy=1:size(A,2), iz=1:size(A,3)].^3);
                            @parallel loopopt=true loopsize=3 function d2_loopopt!(A2, A)
                                @inn(A2) = @d2_yi(A)
                                return
                            end
                            @parallel loopopt=true d2_loopopt!(A2, A);
                            A2_ref[2:end-1,2:end-1,2:end-1] .= A[2:end-1,3:end,2:end-1] .- 2.0.*A[2:end-1,2:end-1,2:end-1] .+ A[2:end-1,1:end-2,2:end-1];
                            @test all(Array(A2) .== Array(A2_ref))
                        end
                        @testset "@parallel_indices <kernel> (3D, loopopt, stencilranges=-1:1)" begin
                            lam=dt=_dx=_dy=_dz = 1.0
                            T      = @zeros(nx, ny, nz);
                            T2     = @zeros(nx, ny, nz);
                            T2_ref = @zeros(nx, ny, nz);
                            Ci     = @ones(nx, ny, nz);
                            copy!(T, [ix + (iy-1)*size(T,1) + (iz-1)*size(T,1)*size(T,2) for ix=1:size(T,1), iy=1:size(T,2), iz=1:size(T,3)].^3);
                            @parallel_indices (ix,iy,iz) loopopt=true loopsize=3 function diffusion3D_step!(T2, T, Ci, lam, dt, _dx, _dy, _dz)
                                if (ix>1 && ix<size(T2,1) && iy>1 && iy<size(T2,2) && iz>1 && iz<size(T2,3))
                                    T2[ix,iy,iz] = T[ix,iy,iz] + dt*(Ci[ix,iy,iz]*(
                                                    - ((-lam*(T[ix+1,iy,iz] - T[ix,iy,iz])*_dx) - (-lam*(T[ix,iy,iz] - T[ix-1,iy,iz])*_dx))*_dx
                                                    - ((-lam*(T[ix,iy+1,iz] - T[ix,iy,iz])*_dy) - (-lam*(T[ix,iy,iz] - T[ix,iy-1,iz])*_dy))*_dy
                                                    - ((-lam*(T[ix,iy,iz+1] - T[ix,iy,iz])*_dz) - (-lam*(T[ix,iy,iz] - T[ix,iy,iz-1])*_dz))*_dz)
                                                    );
                                end
                                return
                            end
                            @parallel loopopt=true diffusion3D_step!(T2, T, Ci, lam, dt, _dx, _dy, _dz);
                            T2_ref[2:end-1,2:end-1,2:end-1] .= T[2:end-1,2:end-1,2:end-1] .+ dt.*(Ci[2:end-1,2:end-1,2:end-1].*(
                                                    - ((.-lam.*(T[3:end  ,2:end-1,2:end-1] .- T[2:end-1,2:end-1,2:end-1]).*_dx) .- (.-lam.*(T[2:end-1,2:end-1,2:end-1] .- T[1:end-2,2:end-1,2:end-1]).*_dx)).*_dx
                                                    - ((.-lam.*(T[2:end-1,3:end  ,2:end-1] .- T[2:end-1,2:end-1,2:end-1]).*_dy) .- (.-lam.*(T[2:end-1,2:end-1,2:end-1] .- T[2:end-1,1:end-2,2:end-1]).*_dy)).*_dy
                                                    - ((.-lam.*(T[2:end-1,2:end-1,3:end  ] .- T[2:end-1,2:end-1,2:end-1]).*_dz) .- (.-lam.*(T[2:end-1,2:end-1,2:end-1] .- T[2:end-1,2:end-1,1:end-2]).*_dz)).*_dz)
                                                    );
                            @test all(Array(T2) .== Array(T2_ref))
                        end
                        @testset "@parallel <kernel> (3D, loopopt, stencilranges=0:2)" begin
                            lam=dt=_dx=_dy=_dz = 1.0
                            T      = @zeros(nx, ny, nz);
                            T2     = @zeros(nx, ny, nz);
                            T2_ref = @zeros(nx, ny, nz);
                            Ci     = @ones(nx, ny, nz);
                            copy!(T, [ix + (iy-1)*size(T,1) + (iz-1)*size(T,1)*size(T,2) for ix=1:size(T,1), iy=1:size(T,2), iz=1:size(T,3)].^3);
                            @parallel loopopt=true loopsize=3 function diffusion3D_step!(T2, T, Ci, lam, dt, _dx, _dy, _dz)
                                @inn(T2) = @inn(T) + dt*(lam*@inn(Ci)*(@d2_xi(T)*_dx^2 + @d2_yi(T)*_dy^2 + @d2_zi(T)*_dz^2))
                                return
                            end
                            @parallel loopopt=true diffusion3D_step!(T2, T, Ci, lam, dt, _dx, _dy, _dz);
                            T2_ref[2:end-1,2:end-1,2:end-1] .= T[2:end-1,2:end-1,2:end-1] .+ dt.*(lam.*Ci[2:end-1,2:end-1,2:end-1].*(
                                                      ((T[3:end  ,2:end-1,2:end-1] .- T[2:end-1,2:end-1,2:end-1]) .- (T[2:end-1,2:end-1,2:end-1] .- T[1:end-2,2:end-1,2:end-1])).*_dx^2
                                                    + ((T[2:end-1,3:end  ,2:end-1] .- T[2:end-1,2:end-1,2:end-1]) .- (T[2:end-1,2:end-1,2:end-1] .- T[2:end-1,1:end-2,2:end-1])).*_dy^2
                                                    + ((T[2:end-1,2:end-1,3:end  ] .- T[2:end-1,2:end-1,2:end-1]) .- (T[2:end-1,2:end-1,2:end-1] .- T[2:end-1,2:end-1,1:end-2])).*_dz^2)
                                                    );
                            @test all(Array(T2) .== Array(T2_ref))
                        end
                        @testset "@parallel_indices <kernel> (3D, loopopt, stencilranges=(-4:-1, 2:2, -2:3); x-z-stencil, y-shift)" begin
                            A      = @zeros(nx, ny, nz);
                            A2     = @zeros(nx, ny, nz);
                            A2_ref = @zeros(nx, ny, nz);
                            copy!(A, [ix + (iy-1)*size(A,1) + (iz-1)*size(A,1)*size(A,2) for ix=1:size(A,1), iy=1:size(A,2), iz=1:size(A,3)].^3);
                            @parallel_indices (ix,iy,iz) loopopt=true loopsize=3 function higher_order_loopopt!(A2, A)
                                if (ix-4>1 && ix-1<size(A2,1) && iy+2>1 && iy+2<=size(A2,2) && iz-2>=1 && iz+3<=size(A2,3))
                                    A2[ix-1,iy+2,iz] = A[ix-1,iy+2,iz+3] - 2.0*A[ix-3,iy+2,iz] + A[ix-4,iy+2,iz-2]
                                end
                                return
                            end
                            @parallel loopopt=true higher_order_loopopt!(A2, A);
                            A2_ref[5:end-1,3:end,3:end-3] .= A[5:end-1,3:end,6:end] .- 2.0.*A[3:end-3,3:end,3:end-3] .+ A[2:end-4,3:end,1:end-5];
                            @test all(Array(A2) .== Array(A2_ref))
                        end
                        @testset "@parallel <kernel> (3D, loopopt, stencilranges=0:0; 2 arrays)" begin
                            A  = @zeros(nx, ny, nz);
                            A2 = @zeros(nx, ny, nz);
                            B  = @zeros(nx, ny, nz);
                            copy!(A, [ix + (iy-1)*size(A,1) + (iz-1)*size(A,1)*size(A,2) for ix=1:size(A,1), iy=1:size(A,2), iz=1:size(A,3)].^3);
                            copy!(B, 2 .* [ix + (iy-1)*size(B,1) + (iz-1)*size(B,1)*size(B,2) for ix=1:size(B,1), iy=1:size(B,2), iz=1:size(B,3)].^3);
                            @parallel loopopt=true loopsize=3 function copy_loopopt!(A2, A, B)
                                @all(A2) = @all(A) + @all(B)
                                return
                            end
                            @parallel loopopt=true copy_loopopt!(A2, A, B);
                            @test all(Array(A2) .== Array(A) .+ Array(B))
                        end
                        @testset "@parallel_indices <kernel> (3D, loopopt; 2 arrays, z-stencil)" begin
                            A      = @zeros(nx, ny, nz);
                            A2     = @zeros(nx, ny, nz);
                            A2_ref = @zeros(nx, ny, nz);
                            B      = @zeros(nx, ny, nz-1);
                            copy!(A, [ix + (iy-1)*size(A,1) + (iz-1)*size(A,1)*size(A,2) for ix=1:size(A,1), iy=1:size(A,2), iz=1:size(A,3)].^3);
                            copy!(B, 2 .* [ix + (iy-1)*size(B,1) + (iz-1)*size(B,1)*size(B,2) for ix=1:size(B,1), iy=1:size(B,2), iz=1:size(B,3)].^3);
                            @parallel_indices (ix,iy,iz) loopopt=true loopsize=3 function d2_loopopt!(A2, A, B)
                                if (iz>1 && iz<size(A2,3))
                                    A2[ix,iy,iz] = A[ix,iy,iz+1] - 2.0*A[ix,iy,iz] + A[ix,iy,iz-1] + B[ix,iy,iz] - B[ix,iy,iz-1]
                                end
                                return
                            end
                            @parallel loopopt=true d2_loopopt!(A2, A, B);
                            A2_ref[:,:,2:end-1] .= A[:,:,3:end] .- 2.0.*A[:,:,2:end-1] .+ A[:,:,1:end-2] .+ B[:,:,2:end] .- B[:,:,1:end-1];
                            @test all(Array(A2) .== Array(A2_ref))
                        end
                        @testset "@parallel_indices <kernel> (3D, loopopt; 2 arrays, y-stencil)" begin
                            A      = @zeros(nx, ny, nz);
                            A2     = @zeros(nx, ny, nz);
                            A2_ref = @zeros(nx, ny, nz);
                            B      = @zeros(nx, ny, nz);
                            copy!(A, [ix + (iy-1)*size(A,1) + (iz-1)*size(A,1)*size(A,2) for ix=1:size(A,1), iy=1:size(A,2), iz=1:size(A,3)].^3);
                            copy!(B, 2 .* [ix + (iy-1)*size(B,1) + (iz-1)*size(B,1)*size(B,2) for ix=1:size(B,1), iy=1:size(B,2), iz=1:size(B,3)].^3);
                            @parallel_indices (ix,iy,iz) loopopt=true loopsize=3 function d2_loopopt!(A2, A, B)
                                if (iy>1 && iy<size(A2,2))
                                    A2[ix,iy,iz] = A[ix,iy+1,iz] - 2.0*A[ix,iy,iz] + A[ix,iy-1,iz] + B[ix,iy+1,iz] - 2.0*B[ix,iy,iz] + B[ix,iy-1,iz]
                                end
                                return
                            end
                            @parallel loopopt=true d2_loopopt!(A2, A, B);
                            A2_ref[:,2:end-1,:] .= (((A[:,3:end,:] .- 2.0.*A[:,2:end-1,:]) .+ A[:,1:end-2,:] .+ B[:,3:end,:]) .- 2.0.*B[:,2:end-1,:]) .+ B[:,1:end-2,:];
                            @test all(Array(A2) .== Array(A2_ref))
                        end
                        @testset "@parallel_indices <kernel> (3D, loopopt; 2 arrays, x-stencil)" begin
                            A      = @zeros(nx, ny, nz);
                            A2     = @zeros(nx, ny, nz);
                            A2_ref = @zeros(nx, ny, nz);
                            B      = @zeros(nx-1, ny, nz);
                            copy!(A, [ix + (iy-1)*size(A,1) + (iz-1)*size(A,1)*size(A,2) for ix=1:size(A,1), iy=1:size(A,2), iz=1:size(A,3)].^3);
                            copy!(B, 2 .* [ix + (iy-1)*size(B,1) + (iz-1)*size(B,1)*size(B,2) for ix=1:size(B,1), iy=1:size(B,2), iz=1:size(B,3)].^3);
                            @parallel_indices (ix,iy,iz) loopopt=true function d2_loopopt!(A2, A, B)
                                if (ix>1 && ix<size(A2,1))
                                    A2[ix,iy,iz] = A[ix+1,iy,iz] - 2.0*A[ix,iy,iz] + A[ix-1,iy,iz] + B[ix,iy,iz] - B[ix-1,iy,iz]
                                end
                                return
                            end
                            @parallel loopopt=true d2_loopopt!(A2, A, B);
                            A2_ref[2:end-1,:,:] .= A[3:end,:,:] .- 2.0.*A[2:end-1,:,:] .+ A[1:end-2,:,:] .+ B[2:end,:,:] .- B[1:end-1,:,:];
                            @test all(Array(A2) .== Array(A2_ref))
                        end
                        @testset "@parallel <kernel> (3D, loopopt; 2 arrays, x-y-z- + z-stencil)" begin
                            lam=dt=_dx=_dy=_dz = 1.0
                            T      = @zeros(nx, ny, nz);
                            T2     = @zeros(nx, ny, nz);
                            T2_ref = @zeros(nx, ny, nz);
                            Ci     = @zeros(nx, ny, nz-1);
                            copy!(T, [ix + (iy-1)*size(T,1) + (iz-1)*size(T,1)*size(T,2) for ix=1:size(T,1), iy=1:size(T,2), iz=1:size(T,3)].^3);
                            copy!(Ci, 2 .* [ix + (iy-1)*size(Ci,1) + (iz-1)*size(Ci,1)*size(Ci,2) for ix=1:size(Ci,1), iy=1:size(Ci,2), iz=1:size(Ci,3)].^3);
                            @parallel loopopt=true loopsize=3 function diffusion3D_step_modified!(T2, T, Ci, lam, dt, _dx, _dy, _dz)
                                @inn(T2) = @inn(T) + dt*(lam*@d_zi(Ci)*(@d2_xi(T)*_dx^2 + @d2_yi(T)*_dy^2 + @d2_zi(T)*_dz^2))
                                return
                            end
                            @parallel loopopt=true diffusion3D_step_modified!(T2, T, Ci, lam, dt, _dx, _dy, _dz);
                            T2_ref[2:end-1,2:end-1,2:end-1] .= T[2:end-1,2:end-1,2:end-1] .+ dt.*(lam.*(Ci[2:end-1,2:end-1,2:end] .- Ci[2:end-1,2:end-1,1:end-1]).*(
                                                      ((T[3:end  ,2:end-1,2:end-1] .- T[2:end-1,2:end-1,2:end-1]) .- (T[2:end-1,2:end-1,2:end-1] .- T[1:end-2,2:end-1,2:end-1])).*_dx^2
                                                    + ((T[2:end-1,3:end  ,2:end-1] .- T[2:end-1,2:end-1,2:end-1]) .- (T[2:end-1,2:end-1,2:end-1] .- T[2:end-1,1:end-2,2:end-1])).*_dy^2
                                                    + ((T[2:end-1,2:end-1,3:end  ] .- T[2:end-1,2:end-1,2:end-1]) .- (T[2:end-1,2:end-1,2:end-1] .- T[2:end-1,2:end-1,1:end-2])).*_dz^2)
                                                    );
                            @test all(Array(T2) .== Array(T2_ref))
                        end
                        @testset "@parallel <kernel> (3D, loopopt; 2 arrays, x-y-z- + x-stencil)" begin
                            lam=dt=_dx=_dy=_dz = 1.0
                            T      = @zeros(nx, ny, nz);
                            T2     = @zeros(nx, ny, nz);
                            T2_ref = @zeros(nx, ny, nz);
                            Ci     = @zeros(nx-1, ny, nz);
                            copy!(T, [ix + (iy-1)*size(T,1) + (iz-1)*size(T,1)*size(T,2) for ix=1:size(T,1), iy=1:size(T,2), iz=1:size(T,3)].^3);
                            copy!(Ci, 2 .* [ix + (iy-1)*size(Ci,1) + (iz-1)*size(Ci,1)*size(Ci,2) for ix=1:size(Ci,1), iy=1:size(Ci,2), iz=1:size(Ci,3)].^3);
                            @parallel loopopt=true loopsize=3 function diffusion3D_step_modified!(T2, T, Ci, lam, dt, _dx, _dy, _dz)
                                @inn(T2) = @inn(T) + dt*(lam*@d_xi(Ci)*(@d2_xi(T)*_dx^2 + @d2_yi(T)*_dy^2 + @d2_zi(T)*_dz^2))
                                return
                            end
                            @parallel loopopt=true diffusion3D_step_modified!(T2, T, Ci, lam, dt, _dx, _dy, _dz);
                            T2_ref[2:end-1,2:end-1,2:end-1] .= T[2:end-1,2:end-1,2:end-1] .+ dt.*(lam.*(Ci[2:end,2:end-1,2:end-1] .- Ci[1:end-1,2:end-1,2:end-1]).*(
                                                      ((T[3:end  ,2:end-1,2:end-1] .- T[2:end-1,2:end-1,2:end-1]) .- (T[2:end-1,2:end-1,2:end-1] .- T[1:end-2,2:end-1,2:end-1])).*_dx^2
                                                    + ((T[2:end-1,3:end  ,2:end-1] .- T[2:end-1,2:end-1,2:end-1]) .- (T[2:end-1,2:end-1,2:end-1] .- T[2:end-1,1:end-2,2:end-1])).*_dy^2
                                                    + ((T[2:end-1,2:end-1,3:end  ] .- T[2:end-1,2:end-1,2:end-1]) .- (T[2:end-1,2:end-1,2:end-1] .- T[2:end-1,2:end-1,1:end-2])).*_dz^2)
                                                    );
                            @test all(Array(T2) .== Array(T2_ref))
                        end
                        @testset "@parallel <kernel> (3D, loopopt; 3 arrays, x-y-z- + y- + x-stencil)" begin
                            lam=dt=_dx=_dy=_dz = 1.0
                            T      = @zeros(nx, ny, nz);
                            T2     = @zeros(nx, ny, nz);
                            T2_ref = @zeros(nx, ny, nz);
                            Ci     = @zeros(nx-1, ny, nz);
                            B      = @zeros(nx, ny, nz);
                            copy!(T, [ix + (iy-1)*size(T,1) + (iz-1)*size(T,1)*size(T,2) for ix=1:size(T,1), iy=1:size(T,2), iz=1:size(T,3)].^3);
                            copy!(Ci, 2 .* [ix + (iy-1)*size(Ci,1) + (iz-1)*size(Ci,1)*size(Ci,2) for ix=1:size(Ci,1), iy=1:size(Ci,2), iz=1:size(Ci,3)].^3);
                            copy!(B,  3 .* [ix + (iy-1)*size(B,1) + (iz-1)*size(B,1)*size(B,2) for ix=1:size(B,1), iy=1:size(B,2), iz=1:size(B,3)].^3);
                            @parallel loopopt=true loopsize=3 function diffusion3D_step_modified!(T2, T, Ci, B, lam, dt, _dx, _dy, _dz)
                                @inn(T2) = @inn(T) + dt*(lam*@d_xi(Ci)*(@d2_xi(T)*_dx^2 + @d2_yi(T)*_dy^2 + @d2_zi(T)*_dz^2)) + @d2_yi(B)
                                return
                            end
                            @parallel loopopt=true diffusion3D_step_modified!(T2, T, Ci, B, lam, dt, _dx, _dy, _dz);
                            T2_ref[2:end-1,2:end-1,2:end-1] .= T[2:end-1,2:end-1,2:end-1] .+ dt.*(lam.*(Ci[2:end,2:end-1,2:end-1] .- Ci[1:end-1,2:end-1,2:end-1]).*(
                                                      ((T[3:end  ,2:end-1,2:end-1] .- T[2:end-1,2:end-1,2:end-1]) .- (T[2:end-1,2:end-1,2:end-1] .- T[1:end-2,2:end-1,2:end-1])).*_dx^2
                                                    + ((T[2:end-1,3:end  ,2:end-1] .- T[2:end-1,2:end-1,2:end-1]) .- (T[2:end-1,2:end-1,2:end-1] .- T[2:end-1,1:end-2,2:end-1])).*_dy^2
                                                    + ((T[2:end-1,2:end-1,3:end  ] .- T[2:end-1,2:end-1,2:end-1]) .- (T[2:end-1,2:end-1,2:end-1] .- T[2:end-1,2:end-1,1:end-2])).*_dz^2)
                                                    ) + ((B[2:end-1,3:end  ,2:end-1] .- B[2:end-1,2:end-1,2:end-1]) .- (B[2:end-1,2:end-1,2:end-1] .- B[2:end-1,1:end-2,2:end-1]));
                            @test all(Array(T2) .== Array(T2_ref))
                        end
                        @testset "@parallel_indices <kernel> (3D, loopopt, stencilranges=(-4:-1, 2:2, -2:3); 3 arrays, x-z-stencil, y-shift)" begin
                            A      = @zeros(nx, ny, nz);
                            A2     = @zeros(nx, ny, nz);
                            A2_ref = @zeros(nx, ny, nz);
                            B      = @zeros(nx, ny, nz);
                            B2     = @zeros(nx, ny, nz);
                            B2_ref = @zeros(nx, ny, nz);
                            C      = @zeros(nx, ny, nz);
                            C2     = @zeros(nx, ny, nz);
                            C2_ref = @zeros(nx, ny, nz);
                            copy!(A, [ix + (iy-1)*size(A,1) + (iz-1)*size(A,1)*size(A,2) for ix=1:size(A,1), iy=1:size(A,2), iz=1:size(A,3)].^3);
                            copy!(B, 2 .* [ix + (iy-1)*size(A,1) + (iz-1)*size(A,1)*size(A,2) for ix=1:size(A,1), iy=1:size(A,2), iz=1:size(A,3)].^3);
                            copy!(C, 3 .* [ix + (iy-1)*size(A,1) + (iz-1)*size(A,1)*size(A,2) for ix=1:size(A,1), iy=1:size(A,2), iz=1:size(A,3)].^3);
                            @parallel_indices (ix,iy,iz) loopopt=true loopsize=3 function higher_order_loopopt!(A2, B2, C2, A, B, C)
                                if (ix-4>1 && ix-1<size(A2,1) && iy+2>1 && iy+2<=size(A2,2) && iz-2>=1 && iz+3<=size(A2,3))
                                    A2[ix-1,iy+2,iz] = A[ix-1,iy+2,iz+3] - 2.0*A[ix-3,iy+2,iz] + A[ix-4,iy+2,iz-2]
                                end
                                if (ix-4>1 && ix-1<size(B2,1) && iy+2>1 && iy+2<=size(B2,2) && iz-2>=1 && iz+3<=size(B2,3))
                                    B2[ix-1,iy+2,iz] = B[ix-1,iy+2,iz+3] - 2.0*B[ix-3,iy+2,iz] + B[ix-4,iy+2,iz-2]
                                end
                                if (ix-4>1 && ix-1<size(C2,1) && iy+2>1 && iy+2<=size(C2,2) && iz-2>=1 && iz+3<=size(C2,3))
                                    C2[ix-1,iy+2,iz] = C[ix-1,iy+2,iz+3] - 2.0*C[ix-3,iy+2,iz] + C[ix-4,iy+2,iz-2]
                                end
                                return
                            end
                            @parallel loopopt=true higher_order_loopopt!(A2, B2, C2, A, B, C);
                            A2_ref[5:end-1,3:end,3:end-3] .= A[5:end-1,3:end,6:end] .- 2.0.*A[3:end-3,3:end,3:end-3] .+ A[2:end-4,3:end,1:end-5];
                            B2_ref[5:end-1,3:end,3:end-3] .= B[5:end-1,3:end,6:end] .- 2.0.*B[3:end-3,3:end,3:end-3] .+ B[2:end-4,3:end,1:end-5];
                            C2_ref[5:end-1,3:end,3:end-3] .= C[5:end-1,3:end,6:end] .- 2.0.*C[3:end-3,3:end,3:end-3] .+ C[2:end-4,3:end,1:end-5];
                            @test all(Array(A2) .== Array(A2_ref))
                            @test all(Array(B2) .== Array(B2_ref))
                            @test all(Array(C2) .== Array(C2_ref))
                        end
                        @testset "@parallel_indices <kernel> (3D, loopopt, stencilranges=(A=(-4:-1, 2:2, -2:3), B=(-4:-1, 2:2, 1:2), C=(-4:-1, 2:2, -1:0)); 3 arrays, x-z-stencil, y-shift)" begin
                            A      = @zeros(nx, ny, nz);
                            A2     = @zeros(nx, ny, nz);
                            A2_ref = @zeros(nx, ny, nz);
                            B      = @zeros(nx, ny, nz);
                            B2     = @zeros(nx, ny, nz);
                            B2_ref = @zeros(nx, ny, nz);
                            C      = @zeros(nx, ny, nz);
                            C2     = @zeros(nx, ny, nz);
                            C2_ref = @zeros(nx, ny, nz);
                            copy!(A, [ix + (iy-1)*size(A,1) + (iz-1)*size(A,1)*size(A,2) for ix=1:size(A,1), iy=1:size(A,2), iz=1:size(A,3)].^3);
                            copy!(B, 2 .* [ix + (iy-1)*size(A,1) + (iz-1)*size(A,1)*size(A,2) for ix=1:size(A,1), iy=1:size(A,2), iz=1:size(A,3)].^3);
                            copy!(C, 3 .* [ix + (iy-1)*size(A,1) + (iz-1)*size(A,1)*size(A,2) for ix=1:size(A,1), iy=1:size(A,2), iz=1:size(A,3)].^3);
                            @parallel_indices (ix,iy,iz) loopopt=true loopsize=3 function higher_order_loopopt!(A2, B2, C2, A, B, C)
                                if (ix-4>1 && ix-1<size(A2,1) && iy+2>1 && iy+2<=size(A2,2) && iz-2>=1 && iz+3<=size(A2,3))
                                    A2[ix-1,iy+2,iz] = A[ix-1,iy+2,iz+3] - 2.0*A[ix-3,iy+2,iz] + A[ix-4,iy+2,iz-2]
                                end
                                if (ix-4>1 && ix-1<size(B2,1) && iy+2>1 && iy+2<=size(B2,2) && iz+1>=1 && iz+2<=size(B2,3))
                                    B2[ix-1,iy+2,iz+1] = B[ix-1,iy+2,iz+2] - 2.0*B[ix-3,iy+2,iz+1] + B[ix-4,iy+2,iz+1]
                                end
                                if (ix-4>1 && ix-1<size(C2,1) && iy+2>1 && iy+2<=size(C2,2) && iz-1>=1 && iz<=size(C2,3))
                                    C2[ix-1,iy+2,iz-1] = C[ix-1,iy+2,iz] - 2.0*C[ix-3,iy+2,iz-1] + C[ix-4,iy+2,iz-1]
                                end
                                return
                            end
                            @parallel loopopt=true higher_order_loopopt!(A2, B2, C2, A, B, C);
                            A2_ref[5:end-1,3:end,3:end-3] .= A[5:end-1,3:end,6:end] .- 2.0.*A[3:end-3,3:end,3:end-3] .+ A[2:end-4,3:end,1:end-5];
                            B2_ref[5:end-1,3:end,2:end-1] .= B[5:end-1,3:end,3:end] .- 2.0.*B[3:end-3,3:end,2:end-1] .+ B[2:end-4,3:end,2:end-1];
                            C2_ref[5:end-1,3:end,1:end-1] .= C[5:end-1,3:end,2:end] .- 2.0.*C[3:end-3,3:end,1:end-1] .+ C[2:end-4,3:end,1:end-1];
                            @test all(Array(A2) .== Array(A2_ref))
                            @test all(Array(B2) .== Array(B2_ref))
                            @test all(Array(C2) .== Array(C2_ref))
                        end
                        @testset "@parallel_indices <kernel> (3D, loopopt, stencilranges=(A=(-4:-1, 2:2, -2:3), B=(-4:1, 2:2, 1:2), C=(-1:-1, 2:2, -1:0)); 3 arrays, x-z-stencil, y-shift)" begin
                            A      = @zeros(nx, ny, nz);
                            A2     = @zeros(nx, ny, nz);
                            A2_ref = @zeros(nx, ny, nz);
                            B      = @zeros(nx, ny, nz);
                            B2     = @zeros(nx, ny, nz);
                            B2_ref = @zeros(nx, ny, nz);
                            C      = @zeros(nx, ny, nz);
                            C2     = @zeros(nx, ny, nz);
                            C2_ref = @zeros(nx, ny, nz);
                            copy!(A, [ix + (iy-1)*size(A,1) + (iz-1)*size(A,1)*size(A,2) for ix=1:size(A,1), iy=1:size(A,2), iz=1:size(A,3)].^3);
                            copy!(B, 2 .* [ix + (iy-1)*size(A,1) + (iz-1)*size(A,1)*size(A,2) for ix=1:size(A,1), iy=1:size(A,2), iz=1:size(A,3)].^3);
                            copy!(C, 3 .* [ix + (iy-1)*size(A,1) + (iz-1)*size(A,1)*size(A,2) for ix=1:size(A,1), iy=1:size(A,2), iz=1:size(A,3)].^3);
                            @parallel_indices (ix,iy,iz) loopopt=true loopsize=3 function higher_order_loopopt!(A2, B2, C2, A, B, C)
                                if (ix-4>1 && ix-1<size(A2,1) && iy+2>1 && iy+2<=size(A2,2) && iz-2>=1 && iz+3<=size(A2,3))
                                    A2[ix-1,iy+2,iz] = A[ix-1,iy+2,iz+3] - 2.0*A[ix-3,iy+2,iz] + A[ix-4,iy+2,iz-2]
                                end
                                if (ix-4>1 && ix+1<size(B2,1) && iy+2>1 && iy+2<=size(B2,2) && iz+1>=1 && iz+2<=size(B2,3))
                                    B2[ix+1,iy+2,iz+1] = B[ix+1,iy+2,iz+2] - 2.0*B[ix-3,iy+2,iz+1] + B[ix-4,iy+2,iz+1]
                                end
                                if (ix-1>1 && ix-1<size(C2,1) && iy+2>1 && iy+2<=size(C2,2) && iz-1>=1 && iz<=size(C2,3))
                                    C2[ix-1,iy+2,iz-1] = C[ix-1,iy+2,iz] - 2.0*C[ix-1,iy+2,iz-1] + C[ix-1,iy+2,iz-1]
                                end
                                return
                            end
                            @parallel loopopt=true higher_order_loopopt!(A2, B2, C2, A, B, C);
                            A2_ref[5:end-1,3:end,3:end-3] .= A[5:end-1,3:end,6:end] .- 2.0.*A[3:end-3,3:end,3:end-3] .+ A[2:end-4,3:end,1:end-5];
                            B2_ref[7:end-1,3:end,2:end-1] .= B[7:end-1,3:end,3:end] .- 2.0.*B[3:end-5,3:end,2:end-1] .+ B[2:end-6,3:end,2:end-1];
                            C2_ref[2:end-1,3:end,1:end-1] .= C[2:end-1,3:end,2:end] .- 2.0.*C[2:end-1,3:end,1:end-1] .+ C[2:end-1,3:end,1:end-1];
                            @test all(Array(A2) .== Array(A2_ref))
                            @test all(Array(B2) .== Array(B2_ref))
                            @test all(Array(C2) .== Array(C2_ref))
                        end
                    end
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
                @static if @isgpu($package)
                    expansion = @prettystring(1, @parallel f(A::Data.Array{T}, B::Data.Array{T}, c::Integer) where T <: PSNumber = (@all(A) = @all(B)^c; return))
                    @test occursin("f(A::Data.DeviceArray{T}, B::Data.DeviceArray{T},", expansion)
                end
            end
            @testset "Data.Cell{T} to Data.DeviceCell{T}" begin
                @static if @isgpu($package)
                    expansion = @prettystring(1, @parallel f(A::Data.Cell{T}, B::Data.Cell{T}, c::Integer) where T <: PSNumber = (@all(A) = @all(B)^c; return))
                    @test occursin("f(A::Data.DeviceCell{T}, B::Data.DeviceCell{T},", expansion)
                end
            end
            @testset "Data.CellArray{T} to Data.DeviceCellArray{T}" begin
                @static if @isgpu($package)
                    expansion = @prettystring(1, @parallel f(A::Data.CellArray{T}, B::Data.CellArray{T}, c::Integer) where T <: PSNumber = (@all(A) = @all(B)^c; return))
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
