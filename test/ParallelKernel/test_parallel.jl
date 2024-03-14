using Test
import ParallelStencil
using Enzyme
using ParallelStencil.ParallelKernel
import ParallelStencil.ParallelKernel.AD
import ParallelStencil.ParallelKernel: @reset_parallel_kernel, @is_initialized, SUPPORTED_PACKAGES, PKG_CUDA, PKG_AMDGPU, PKG_THREADS, INDICES
import ParallelStencil.ParallelKernel: @require, @prettystring, @gorgeousstring, @isgpu
import ParallelStencil.ParallelKernel: checkargs_parallel, checkargs_parallel_indices, parallel_indices, maxsize
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
Base.retry_load_extensions() # Potentially needed to load the extensions after the packages have been filtered.

macro compute(A)              esc(:($(INDICES[1]) + ($(INDICES[2])-1)*size($A,1))) end
macro compute_with_aliases(A) esc(:(ix            + (iz           -1)*size($A,1))) end
import Enzyme
@static for package in TEST_PACKAGES  eval(:(
    @testset "$(basename(@__FILE__)) (package: $(nameof($package)))" begin
        @testset "1. parallel macros" begin
            @require !@is_initialized()
            @init_parallel_kernel($package, Float64)
            @require @is_initialized()
            @testset "@parallel" begin
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
                elseif $package == $PKG_AMDGPU
                    call = @prettystring(1, @parallel f(A))
                    @test occursin("AMDGPU.@roc gridsize = ParallelStencil.ParallelKernel.compute_nblocks(length.(ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A))), ParallelStencil.ParallelKernel.compute_nthreads(length.(ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A))))) groupsize = ParallelStencil.ParallelKernel.compute_nthreads(length.(ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A)))) stream = AMDGPU.stream() f(A, ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A)), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A)))[1])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A)))[2])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A)))[3])))", call)
                    @test occursin("AMDGPU.synchronize(AMDGPU.stream())", call)
                    call = @prettystring(1, @parallel ranges f(A))
                    @test occursin("AMDGPU.@roc gridsize = ParallelStencil.ParallelKernel.compute_nblocks(length.(ParallelStencil.ParallelKernel.promote_ranges(ranges)), ParallelStencil.ParallelKernel.compute_nthreads(length.(ParallelStencil.ParallelKernel.promote_ranges(ranges)))) groupsize = ParallelStencil.ParallelKernel.compute_nthreads(length.(ParallelStencil.ParallelKernel.promote_ranges(ranges))) stream = AMDGPU.stream() f(A, ParallelStencil.ParallelKernel.promote_ranges(ranges), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ranges))[1])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ranges))[2])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ranges))[3])))", call)
                    call = @prettystring(1, @parallel nblocks nthreads f(A))
                    @test occursin("AMDGPU.@roc gridsize = nblocks groupsize = nthreads stream = AMDGPU.stream() f(A, ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads)), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads)))[1])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads)))[2])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads)))[3])))", call)
                    call = @prettystring(1, @parallel ranges nblocks nthreads f(A))
                    @test occursin("AMDGPU.@roc gridsize = nblocks groupsize = nthreads stream = AMDGPU.stream() f(A, ParallelStencil.ParallelKernel.promote_ranges(ranges), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ranges))[1])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ranges))[2])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ranges))[3])))", call)
                    call = @prettystring(1, @parallel nblocks nthreads stream=mystream f(A))
                    @test occursin("AMDGPU.@roc gridsize = nblocks groupsize = nthreads stream = mystream f(A, ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads)), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads)))[1])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads)))[2])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads)))[3])))", call)
                elseif $package == $PKG_THREADS
                    @test @prettystring(1, @parallel f(A)) == "f(A, ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A)), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A)))[1])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A)))[2])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A)))[3])))"
                    @test @prettystring(1, @parallel ranges f(A)) == "f(A, ParallelStencil.ParallelKernel.promote_ranges(ranges), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ranges))[1])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ranges))[2])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ranges))[3])))"
                    @test @prettystring(1, @parallel nblocks nthreads f(A)) == "f(A, ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads)), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads)))[1])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads)))[2])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads)))[3])))"
                    @test @prettystring(1, @parallel ranges nblocks nthreads f(A)) == "f(A, ParallelStencil.ParallelKernel.promote_ranges(ranges), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ranges))[1])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ranges))[2])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ranges))[3])))"
                    @test @prettystring(1, @parallel stream=mystream f(A)) == "f(A, ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A)), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A)))[1])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A)))[2])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A)))[3])))"
                end;
                call = @prettystring(1, @parallel configcall=g(B) f(A))
                @test  occursin("get_ranges(B)", call)
                @test !occursin("get_ranges(A)", call)
                @test  occursin("f(A,", call)
                @test !occursin("g(B,", call)
                @testset "maxsize" begin
                    struct BitstypeStruct
                        x::Int
                        y::Float64
                    end
                    @test maxsize([9 9; 9 9; 9 9]) == (3, 2, 1)
                    @test maxsize(8) == (1, 1, 1)
                    @test maxsize(BitstypeStruct(5, 6.0)) == (1, 1, 1)
                    @test maxsize([9 9; 9 9; 9 9], [7 7 7; 7 7 7]) == (3, 3, 1)
                    @test maxsize(8, [9 9; 9 9; 9 9], [7 7 7; 7 7 7]) == (3, 3, 1)
                    @test maxsize(BitstypeStruct(5, 6.0), 8, [9 9; 9 9; 9 9], [7 7 7; 7 7 7]) == (3, 3, 1)
                    @test maxsize((x=8, y=[9 9; 9 9; 9 9], z=[7 7 7; 7 7 7])) == (3, 3, 1)
                    @test maxsize((x=8, y=[9 9; 9 9; 9 9]), [7 7 7; 7 7 7]) == (3, 3, 1)
                    @test maxsize(BitstypeStruct(5, 6.0), 8, (x=[9 9; 9 9; 9 9], y=[9 9; 9 9; 9 9]), (x=[7 7 7; 7 7 7], y=[7 7 7; 7 7 7])) == (3, 3, 1)
                end;
            end;
            @testset "@parallel ∇" begin
                @test @prettystring(1, @parallel ∇=B->B̄ f!(A, B, a)) == "@parallel configcall = f!(A, B, a) ParallelStencil.ParallelKernel.AD.autodiff_deferred!(Enzyme.Reverse, f!, Enzyme.Const(A), Enzyme.DuplicatedNoNeed(B, B̄), Enzyme.Const(a))"
                @test @prettystring(1, @parallel ∇=(A->Ā, B->B̄) f!(A, B, a)) == "@parallel configcall = f!(A, B, a) ParallelStencil.ParallelKernel.AD.autodiff_deferred!(Enzyme.Reverse, f!, Enzyme.DuplicatedNoNeed(A, Ā), Enzyme.DuplicatedNoNeed(B, B̄), Enzyme.Const(a))"
                @test @prettystring(1, @parallel ∇=(A->Ā, B->B̄) ad_mode=Enzyme.Forward f!(A, B, a)) == "@parallel configcall = f!(A, B, a) ParallelStencil.ParallelKernel.AD.autodiff_deferred!(Enzyme.Forward, f!, Enzyme.DuplicatedNoNeed(A, Ā), Enzyme.DuplicatedNoNeed(B, B̄), Enzyme.Const(a))"
                @test @prettystring(1, @parallel ∇=(A->Ā, B->B̄) ad_mode=Enzyme.Forward ad_annotations=(Duplicated=B) f!(A, B, a)) == "@parallel configcall = f!(A, B, a) ParallelStencil.ParallelKernel.AD.autodiff_deferred!(Enzyme.Forward, f!, Enzyme.DuplicatedNoNeed(A, Ā), Enzyme.Duplicated(B, B̄), Enzyme.Const(a))"
                @test @prettystring(1, @parallel ∇=(A->Ā, B->B̄) ad_mode=Enzyme.Forward ad_annotations=(Duplicated=(B,A), Active=b) f!(A, B, a, b)) == "@parallel configcall = f!(A, B, a, b) ParallelStencil.ParallelKernel.AD.autodiff_deferred!(Enzyme.Forward, f!, Enzyme.Duplicated(A, Ā), Enzyme.Duplicated(B, B̄), Enzyme.Const(a), Enzyme.Active(b))"
                @test @prettystring(1, @parallel ∇=(V.x->V̄.x, V.y->V̄.y) f!(V.x, V.y, a)) == "@parallel configcall = f!(V.x, V.y, a) ParallelStencil.ParallelKernel.AD.autodiff_deferred!(Enzyme.Reverse, f!, Enzyme.DuplicatedNoNeed(V.x, V̄.x), Enzyme.DuplicatedNoNeed(V.y, V̄.y), Enzyme.Const(a))"
            end;
            @testset "AD.autodiff_deferred!" begin
                @static if $package == $PKG_THREADS
                    N = 16
                    a = 6.5
                    A = @rand(N)
                    B = @rand(N)
                    Ā = @ones(N)
                    B̄ = @ones(N)
                    A_ref = Array(A)
                    B_ref = Array(B)
                    Ā_ref = ones(N)
                    B̄_ref = ones(N)
                    @parallel_indices (ix) function f!(A, B, a)
                        A[ix] += a * B[ix] * 100.65
                        return
                    end
                    function g!(A, B, a)
                        for ix in 1:length(A)
                            A[ix] += a * B[ix] * 100.65
                        end
                        return
                    end
                    @parallel configcall=f!(A, B, a) AD.autodiff_deferred!(Enzyme.Reverse, f!, DuplicatedNoNeed(A, Ā), DuplicatedNoNeed(B, B̄), Const(a))
                    Enzyme.autodiff_deferred(Enzyme.Reverse, g!, DuplicatedNoNeed(A_ref, Ā_ref), DuplicatedNoNeed(B_ref, B̄_ref), Const(a))
                    @test Array(Ā) ≈ Ā_ref
                    @test Array(B̄) ≈ B̄_ref
                end
            end
            @testset "@parallel_indices" begin
                @testset "inbounds" begin
                    expansion = @prettystring(1, @parallel_indices (ix) inbounds=true f(A) = (2*A; return))
                    @test occursin("Base.@inbounds begin", expansion)
                    expansion = @prettystring(1, @parallel_indices (ix) inbounds=false f(A) = (2*A; return))
                    @test !occursin("Base.@inbounds begin", expansion)
                    expansion = @prettystring(1, @parallel_indices (ix) f(A) = (2*A; return))
                    @test !occursin("Base.@inbounds begin", expansion)
                end
                @testset "addition of range arguments" begin
                    expansion = @gorgeousstring(1, @parallel_indices (ix,iy) f(a::T, b::T) where T <: Union{Array{Float32}, Array{Float64}} = (println("a=$a, b=$b)"); return))
                    @test occursin("f(a::T, b::T, ranges::Tuple{UnitRange, UnitRange, UnitRange}, rangelength_x::Int64, rangelength_y::Int64, rangelength_z::Int64", expansion)
                end
                @testset "Data.Array to Data.DeviceArray" begin
                    @static if @isgpu($package)
                            expansion = @prettystring(1, @parallel_indices (ix,iy) f(A::Data.Array, B::Data.Array, c::T) where T <: Integer = (A[ix,iy] = B[ix,iy]^c; return))
                            @test occursin("f(A::Data.DeviceArray, B::Data.DeviceArray,", expansion)
                    end
                end
                @testset "Data.Cell to Data.DeviceCell" begin
                    @static if @isgpu($package)
                            expansion = @prettystring(1, @parallel_indices (ix,iy) f(A::Data.Cell, B::Data.Cell, c::T) where T <: Integer = (A[ix,iy] = B[ix,iy]^c; return))
                            @test occursin("f(A::Data.DeviceCell, B::Data.DeviceCell,", expansion)
                    end
                end
                @testset "Data.CellArray to Data.DeviceCellArray" begin
                    @static if @isgpu($package)
                            expansion = @prettystring(1, @parallel_indices (ix,iy) f(A::Data.CellArray, B::Data.CellArray, c::T) where T <: Integer = (A[ix,iy] = B[ix,iy]^c; return))
                            @test occursin("f(A::Data.DeviceCellArray, B::Data.DeviceCellArray,", expansion)
                    end
                end
                @testset "Data.ArrayTuple to Data.DeviceArrayTuple" begin
                    @static if @isgpu($package)
                            expansion = @prettystring(1, @parallel_indices (ix,iy) f(A::Data.ArrayTuple, B::Data.ArrayTuple, c::T) where T <: Integer = return)
                            @test occursin("f(A::Data.DeviceArrayTuple, B::Data.DeviceArrayTuple,", expansion)
                    end
                end
                @testset "Data.CellTuple to Data.DeviceCellTuple" begin
                    @static if @isgpu($package)
                            expansion = @prettystring(1, @parallel_indices (ix,iy) f(A::Data.CellTuple, B::Data.CellTuple, c::T) where T <: Integer = return)
                            @test occursin("f(A::Data.DeviceCellTuple, B::Data.DeviceCellTuple,", expansion)
                    end
                end
                @testset "Data.CellArrayTuple to Data.DeviceCellArrayTuple" begin
                    @static if @isgpu($package)
                            expansion = @prettystring(1, @parallel_indices (ix,iy) f(A::Data.CellArrayTuple, B::Data.CellArrayTuple, c::T) where T <: Integer = return)
                            @test occursin("f(A::Data.DeviceCellArrayTuple, B::Data.DeviceCellArrayTuple,", expansion)
                    end
                end
                @testset "Data.NamedArrayTuple to Data.NamedDeviceArrayTuple" begin
                    @static if @isgpu($package)
                            expansion = @prettystring(1, @parallel_indices (ix,iy) f(A::Data.NamedArrayTuple, B::Data.NamedArrayTuple, c::T) where T <: Integer = return)
                            @test occursin("f(A::Data.NamedDeviceArrayTuple, B::Data.NamedDeviceArrayTuple,", expansion)
                    end
                end
                @testset "Data.NamedCellTuple to Data.NamedDeviceCellTuple" begin
                    @static if @isgpu($package)
                            expansion = @prettystring(1, @parallel_indices (ix,iy) f(A::Data.NamedCellTuple, B::Data.NamedCellTuple, c::T) where T <: Integer = return)
                            @test occursin("f(A::Data.NamedDeviceCellTuple, B::Data.NamedDeviceCellTuple,", expansion)
                    end
                end
                @testset "Data.NamedCellArrayTuple to Data.NamedDeviceCellArrayTuple" begin
                    @static if @isgpu($package)
                            expansion = @prettystring(1, @parallel_indices (ix,iy) f(A::Data.NamedCellArrayTuple, B::Data.NamedCellArrayTuple, c::T) where T <: Integer = return)
                            @test occursin("f(A::Data.NamedDeviceCellArrayTuple, B::Data.NamedDeviceCellArrayTuple,", expansion)
                    end
                end
                @testset "Data.ArrayCollection to Data.DeviceArrayCollection" begin
                    @static if @isgpu($package)
                            expansion = @prettystring(1, @parallel_indices (ix,iy) f!(A::Data.ArrayCollection, B::Data.ArrayCollection, c::T) where T <: Integer = (A[ix,iy] = B[ix,iy]^c; return))
                            @test occursin("f!(A::Data.DeviceArrayCollection, B::Data.DeviceArrayCollection,", expansion)
                    end
                end
                @testset "Data.CellCollection to Data.DeviceCellCollection" begin
                    @static if @isgpu($package)
                            expansion = @prettystring(1, @parallel_indices (ix,iy) f!(A::Data.CellCollection, B::Data.CellCollection, c::T) where T <: Integer = (A[ix,iy] = B[ix,iy]^c; return))
                            @test occursin("f!(A::Data.DeviceCellCollection, B::Data.DeviceCellCollection,", expansion)
                    end
                end
                @testset "Data.CellArrayCollection to Data.DeviceCellArrayCollection" begin
                    @static if @isgpu($package)
                            expansion = @prettystring(1, @parallel_indices (ix,iy) f!(A::Data.CellArrayCollection, B::Data.CellArrayCollection, c::T) where T <: Integer = (A[ix,iy] = B[ix,iy]^c; return))
                            @test occursin("f!(A::Data.DeviceCellArrayCollection, B::Data.DeviceCellArrayCollection,", expansion)
                    end
                end
                @testset "Data.TArray to Data.DeviceTArray" begin
                    @static if @isgpu($package)
                            expansion = @prettystring(1, @parallel_indices (ix,iy) f(A::Data.TArray, B::Data.TArray, c::T) where T <: Integer = (A[ix,iy] = B[ix,iy]^c; return))
                            @test occursin("f(A::Data.DeviceTArray, B::Data.DeviceTArray,", expansion)
                    end
                end
                @testset "Data.TCell to Data.DeviceTCell" begin
                    @static if @isgpu($package)
                            expansion = @prettystring(1, @parallel_indices (ix,iy) f(A::Data.TCell, B::Data.TCell, c::T) where T <: Integer = (A[ix,iy] = B[ix,iy]^c; return))
                            @test occursin("f(A::Data.DeviceTCell, B::Data.DeviceTCell,", expansion)
                    end
                end
                @testset "Data.TCellArray to Data.DeviceTCellArray" begin
                    @static if @isgpu($package)
                            expansion = @prettystring(1, @parallel_indices (ix,iy) f(A::Data.TCellArray, B::Data.TCellArray, c::T) where T <: Integer = (A[ix,iy] = B[ix,iy]^c; return))
                            @test occursin("f(A::Data.DeviceTCellArray, B::Data.DeviceTCellArray,", expansion)
                    end
                end
                @testset "Data.TArrayTuple to Data.DeviceTArrayTuple" begin
                    @static if @isgpu($package)
                            expansion = @prettystring(1, @parallel_indices (ix,iy) f(A::Data.TArrayTuple, B::Data.TArrayTuple, c::T) where T <: Integer = return)
                            @test occursin("f(A::Data.DeviceTArrayTuple, B::Data.DeviceTArrayTuple,", expansion)
                    end
                end
                @testset "Data.TCellTuple to Data.DeviceTCellTuple" begin
                    @static if @isgpu($package)
                            expansion = @prettystring(1, @parallel_indices (ix,iy) f(A::Data.TCellTuple, B::Data.TCellTuple, c::T) where T <: Integer = return)
                            @test occursin("f(A::Data.DeviceTCellTuple, B::Data.DeviceTCellTuple,", expansion)
                    end
                end
                @testset "Data.TCellArrayTuple to Data.DeviceTCellArrayTuple" begin
                    @static if @isgpu($package)
                            expansion = @prettystring(1, @parallel_indices (ix,iy) f(A::Data.TCellArrayTuple, B::Data.TCellArrayTuple, c::T) where T <: Integer = return)
                            @test occursin("f(A::Data.DeviceTCellArrayTuple, B::Data.DeviceTCellArrayTuple,", expansion)
                    end
                end
                @testset "Data.NamedTArrayTuple to Data.NamedDeviceTArrayTuple" begin
                    @static if @isgpu($package)
                            expansion = @prettystring(1, @parallel_indices (ix,iy) f(A::Data.NamedTArrayTuple, B::Data.NamedTArrayTuple, c::T) where T <: Integer = return)
                            @test occursin("f(A::Data.NamedDeviceTArrayTuple, B::Data.NamedDeviceTArrayTuple,", expansion)
                    end
                end
                @testset "Data.NamedTCellTuple to Data.NamedDeviceTCellTuple" begin
                    @static if @isgpu($package)
                            expansion = @prettystring(1, @parallel_indices (ix,iy) f(A::Data.NamedTCellTuple, B::Data.NamedTCellTuple, c::T) where T <: Integer = return)
                            @test occursin("f(A::Data.NamedDeviceTCellTuple, B::Data.NamedDeviceTCellTuple,", expansion)
                    end
                end
                @testset "Data.NamedTCellArrayTuple to Data.NamedDeviceTCellArrayTuple" begin
                    @static if @isgpu($package)
                            expansion = @prettystring(1, @parallel_indices (ix,iy) f(A::Data.NamedTCellArrayTuple, B::Data.NamedTCellArrayTuple, c::T) where T <: Integer = return)
                            @test occursin("f(A::Data.NamedDeviceTCellArrayTuple, B::Data.NamedDeviceTCellArrayTuple,", expansion)
                    end
                end
                @testset "Data.TArrayCollection to Data.DeviceTArrayCollection" begin
                    @static if @isgpu($package)
                            expansion = @prettystring(1, @parallel_indices (ix,iy) f!(A::Data.TArrayCollection, B::Data.TArrayCollection, c::T) where T <: Integer = (A[ix,iy] = B[ix,iy]^c; return))
                            @test occursin("f!(A::Data.DeviceTArrayCollection, B::Data.DeviceTArrayCollection,", expansion)
                    end
                end
                @testset "Data.TCellCollection to Data.DeviceTCellCollection" begin
                    @static if @isgpu($package)
                            expansion = @prettystring(1, @parallel_indices (ix,iy) f!(A::Data.TCellCollection, B::Data.TCellCollection, c::T) where T <: Integer = (A[ix,iy] = B[ix,iy]^c; return))
                            @test occursin("f!(A::Data.DeviceTCellCollection, B::Data.DeviceTCellCollection,", expansion)
                    end
                end
                @testset "Data.TCellArrayCollection to Data.DeviceTCellArrayCollection" begin
                    @static if @isgpu($package)
                            expansion = @prettystring(1, @parallel_indices (ix,iy) f!(A::Data.TCellArrayCollection, B::Data.TCellArrayCollection, c::T) where T <: Integer = (A[ix,iy] = B[ix,iy]^c; return))
                            @test occursin("f!(A::Data.DeviceTCellArrayCollection, B::Data.DeviceTCellArrayCollection,", expansion)
                    end
                end
                @testset "Nested Data.Array to Data.DeviceArray" begin
                    @static if @isgpu($package)
                            expansion = @prettystring(1, @parallel_indices (ix,iy) f(A::NamedTuple{T1, NTuple{T2,T3}} where {T1,T2} where T3 <: Data.Array, c::T) where T <: Integer = (A[ix,iy] = B[ix,iy]^c; return))
                            @test occursin("f(A::((NamedTuple{T1, NTuple{T2, T3}} where {T1, T2}) where T3 <: Data.DeviceArray),", expansion)
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
                @testset "@parallel_indices (2D in 3D with macro)" begin
                    A  = @zeros(4, 5, 6)
                    @parallel_indices (ix,iz) function write_indices!(A)
                        A[ix,end,iz] = @compute(A);
                        return
                    end
                    @parallel (1:size(A,1), 1:size(A,3)) write_indices!(A);
                    @test all(Array(A)[:,end,:] .== [ix + (iz-1)*size(A,1) for ix=1:size(A,1), iz=1:size(A,3)])
                end;
                @testset "@parallel_indices (2D in 3D with macro with aliases)" begin
                    A  = @zeros(4, 5, 6)
                    @parallel_indices (ix,iz) function write_indices!(A)
                        A[ix,end,iz] = @compute_with_aliases(A);
                        return
                    end
                    @parallel (1:size(A,1), 1:size(A,3)) write_indices!(A);
                    @test all(Array(A)[:,end,:] .== [ix + (iz-1)*size(A,1) for ix=1:size(A,1), iz=1:size(A,3)])
                end;
                @testset "nested function (long definition, array modification)" begin
                    A  = @zeros(4, 5, 6)
                    @parallel_indices (ix,iy,iz) function write_indices!(A)
                        function compute_indices!(A)
                            A[ix,iy,iz] = ix + (iy-1)*size(A,1) + (iz-1)*size(A,1)*size(A,2);
                            return
                        end
                        compute_indices!(A)
                        return
                    end
                    @parallel write_indices!(A);
                    @test all(Array(A) .== [ix + (iy-1)*size(A,1) + (iz-1)*size(A,1)*size(A,2) for ix=1:size(A,1), iy=1:size(A,2), iz=1:size(A,3)])
                end;
                @testset "nested function (short definition, array modification)" begin
                    A  = @zeros(4, 5, 6)
                    @parallel_indices (ix,iy,iz) function write_indices!(A)
                        compute_indices!(A) = (A[ix,iy,iz] = ix + (iy-1)*size(A,1) + (iz-1)*size(A,1)*size(A,2); return)
                        compute_indices!(A)
                        return
                    end
                    @parallel write_indices!(A);
                    @test all(Array(A) .== [ix + (iy-1)*size(A,1) + (iz-1)*size(A,1)*size(A,2) for ix=1:size(A,1), iy=1:size(A,2), iz=1:size(A,3)])
                end;
                @testset "nested function (long definition, return value)" begin
                    A  = @zeros(4, 5, 6)
                    @parallel_indices (ix,iy,iz) function write_indices!(A)
                        function compute_indices(A)
                            return ix + (iy-1)*size(A,1) + (iz-1)*size(A,1)*size(A,2)
                        end
                        A[ix,iy,iz] = compute_indices(A)
                        return
                    end
                    @parallel write_indices!(A);
                    @test all(Array(A) .== [ix + (iy-1)*size(A,1) + (iz-1)*size(A,1)*size(A,2) for ix=1:size(A,1), iy=1:size(A,2), iz=1:size(A,3)])
                end;
                @testset "nested function (short definition, return value)" begin
                    A  = @zeros(4, 5, 6)
                    @parallel_indices (ix,iy,iz) function write_indices!(A)
                        compute_indices(A) = return ix + (iy-1)*size(A,1) + (iz-1)*size(A,1)*size(A,2)
                        A[ix,iy,iz] = compute_indices(A)
                        return
                    end
                    @parallel write_indices!(A);
                    @test all(Array(A) .== [ix + (iy-1)*size(A,1) + (iz-1)*size(A,1)*size(A,2) for ix=1:size(A,1), iy=1:size(A,2), iz=1:size(A,3)])
                end;
            end;
            @testset "@parallel_async" begin
                @static if @isgpu($package)
                    call = @prettystring(1, @parallel_async f(A))
                    @test !occursin("synchronize", call)
                end;
            end;
            @testset "@synchronize" begin
                @static if $package == $PKG_CUDA
                    @test @prettystring(1, @synchronize()) == "CUDA.synchronize()"
                    @test @prettystring(1, @synchronize(mystream)) == "CUDA.synchronize(mystream)"
                elseif $package == $PKG_AMDGPU
                    @test @prettystring(1, @synchronize()) == "AMDGPU.synchronize()"
                    @test @prettystring(1, @synchronize(mystream)) == "AMDGPU.synchronize(mystream)"
                end;
            end;
            @reset_parallel_kernel()
        end;
        @testset "2. parallel macros (literal conversion)" begin
            @testset "@parallel_indices (Float64)" begin
                @require !@is_initialized()
                @init_parallel_kernel($package, Float64)
                @require @is_initialized()
                expansion = @gorgeousstring(@parallel_indices (ix) f!(A) = (A[ix] = A[ix] + 1.0f0; return))
                @test occursin("A[ix] = A[ix] + 1.0\n", expansion)
                @reset_parallel_kernel()
            end;
            @testset "@parallel_indices (Float32)" begin
                @require !@is_initialized()
                @init_parallel_kernel($package, Float32)
                @require @is_initialized()
                expansion = @gorgeousstring(@parallel_indices (ix) f!(A) = (A[ix] = A[ix] + 1.0; return))
                @test occursin("A[ix] = A[ix] + 1.0f0\n", expansion)
                @reset_parallel_kernel()
            end;
            @testset "@parallel_indices (Float16)" begin
                @require !@is_initialized()
                @init_parallel_kernel($package, Float16)
                @require @is_initialized()
                expansion = @gorgeousstring(@parallel_indices (ix) f!(A) = (A[ix] = A[ix] + 1.0; return))
                @test occursin("A[ix] = A[ix] + Float16(1.0)\n", expansion)
                @reset_parallel_kernel()
            end;
            @testset "@parallel_indices (ComplexF64)" begin
                @require !@is_initialized()
                @init_parallel_kernel($package, ComplexF64)
                @require @is_initialized()
                expansion = @gorgeousstring(@parallel_indices (ix) f!(A) = (A[ix] = 2.0f0 - 1.0f0im - A[ix] + 1.0f0; return))
                @test occursin("A[ix] = ((2.0 - 1.0im) - A[ix]) + 1.0\n", expansion)
                @reset_parallel_kernel()
            end;
            @testset "@parallel_indices (ComplexF32)" begin
                @require !@is_initialized()
                @init_parallel_kernel($package, ComplexF32)
                @require @is_initialized()
                expansion = @gorgeousstring(@parallel_indices (ix) f!(A) = (A[ix] = 2.0 - 1.0im - A[ix] + 1.0; return))
                @test occursin("A[ix] = ((2.0f0 - 1.0f0im) - A[ix]) + 1.0f0\n", expansion)
                @reset_parallel_kernel()
            end;
            @testset "@parallel_indices (ComplexF16)" begin
                @require !@is_initialized()
                @init_parallel_kernel($package, ComplexF16)
                @require @is_initialized()
                expansion = @gorgeousstring(@parallel_indices (ix) f!(A) = (A[ix] = 2.0 - 1.0im - A[ix] + 1.0; return))
                @test occursin("A[ix] = ((Float16(2.0) - Float16(1.0) * im) - A[ix]) + Float16(1.0)\n", expansion)
                @reset_parallel_kernel()
            end;
        end;
        @testset "3. global defaults" begin
            @testset "inbounds=true" begin
                @require !@is_initialized()
                @init_parallel_kernel($package, Float64, inbounds=true)
                @require @is_initialized
                expansion = @prettystring(1, @parallel_indices (ix) inbounds=true f(A) = (2*A; return))
                @test occursin("Base.@inbounds begin", expansion)
                expansion = @prettystring(1, @parallel_indices (ix) f(A) = (2*A; return))
                @test occursin("Base.@inbounds begin", expansion)
                expansion = @prettystring(1, @parallel_indices (ix) inbounds=false f(A) = (2*A; return))
                @test !occursin("Base.@inbounds begin", expansion)
                @reset_parallel_kernel()
            end;
        end;
        @testset "4. parallel macros (numbertype ommited)" begin
            @require !@is_initialized()
            @init_parallel_kernel(package = $package)
            @require @is_initialized
            @testset "Data.Array{T} to Data.DeviceArray{T}" begin
                @static if @isgpu($package)
                        expansion = @prettystring(1, @parallel_indices (ix,iy) f(A::Data.Array{T}, B::Data.Array{T}, c<:Integer) where T <: Union{Float32, Float64}  = (A[ix,iy] = B[ix,iy]^c; return))
                        @test occursin("f(A::Data.DeviceArray{T}, B::Data.DeviceArray{T},", expansion)
                end
            end;
            @testset "Data.Cell{T} to Data.DeviceCell{T}" begin
                @static if @isgpu($package)
                        expansion = @prettystring(1, @parallel_indices (ix,iy) f(A::Data.Cell{T}, B::Data.Cell{T}, c<:Integer) where T <: Union{Float32, Float64}  = (A[ix,iy] = B[ix,iy]^c; return))
                        @test occursin("f(A::Data.DeviceCell{T}, B::Data.DeviceCell{T},", expansion)
                end
            end;
            @testset "Data.CellArray{T} to Data.DeviceCellArray{T}" begin
                @static if @isgpu($package)
                        expansion = @prettystring(1, @parallel_indices (ix,iy) f(A::Data.CellArray{T}, B::Data.CellArray{T}, c<:Integer) where T <: Union{Float32, Float64}  = (A[ix,iy] = B[ix,iy]^c; return))
                        @test occursin("f(A::Data.DeviceCellArray{T}, B::Data.DeviceCellArray{T},", expansion)
                end
            end;
            @reset_parallel_kernel()
        end;
        @testset "5. Exceptions" begin
            @require !@is_initialized()
            @init_parallel_kernel($package, Float64)
            @require @is_initialized
            @testset "arguments @parallel" begin
                @test_throws ArgumentError checkargs_parallel();                                                        # Error: isempty(args)
                @test_throws ArgumentError checkargs_parallel(:(f()), :(something));                                    # Error: last arg is not function call.
                @test_throws ArgumentError checkargs_parallel(:(f()=99));                                               # Error: last arg is not function call.
                #TODO: kw for calls look very different: head :kw - fix in parallel.jl/shared.jl
                @test_throws ArgumentError checkargs_parallel(:(f(;s=1)));                                              # Error: function call with keyword argument.
                @test_throws ArgumentError checkargs_parallel(:ranges, :nblocks, :nthreads, :something, :(f()));        # Error: length(posargs) > 3
                @test_throws KeywordArgumentError checkargs_parallel(:(blocks=blocks), :(f()));                         # Error: blocks keyword argument is not allowed
                @test_throws KeywordArgumentError checkargs_parallel(:(threads=threads), :(f()));                       # Error: threads keyword argument is not allowed
            end;
            @testset "arguments @parallel_indices" begin
                @test_throws ArgumentError checkargs_parallel_indices();                                                # Error: length(args) != 2
                @test_throws ArgumentError checkargs_parallel_indices(:(f()=99));                                       # Error: length(args) != 2
                @test_throws ArgumentError checkargs_parallel_indices(:((ix,iy,iz)), :(f()=99), :(something));          # Error: length(args) != 2
                @test_throws ArgumentError checkargs_parallel_indices(:ix, :iy, :iz, :(f()=99));                        # Error: length(args) != 2
                @test_throws ArgumentError checkargs_parallel_indices(:(f()=99), :((ix,iy,iz)));                        # Error: last arg is not function.
                @test_throws ArgumentError checkargs_parallel_indices(:((ix,iy,iz)), :(f()));                           # Error: last arg is not function.
                @test_throws ArgumentError checkargs_parallel_indices(:((ix,iy,iz)), :(f(;s=1)=(99*s; return)))         # Error: function defines keyword.
                @test_throws ArgumentError parallel_indices(@__MODULE__, :((ix,iy,iz)), :(f()=99))                      # Error: no return statement in function.
                @test_throws ArgumentError parallel_indices(@__MODULE__, :((ix,iy,iz)), :(f()=(99; return something)))  # Error: function does not return nothing.
                #TODO: this tests does not pass anymore for unknown reasons:
                #@test_throws ArgumentError parallel_indices(:((ix,iy,iz)), :(f()=(99; if x return y end; return)))  # Error: function contains more than one return statement.
            end;
            @testset "maxsize" begin
                struct NonBitstypeStruct
                    x::Int
                    y::Array
                end
                @test_throws ArgumentError maxsize(NonBitstypeStruct(5, [6.0]));                                        # Error: argument is not a bitstype.
            end;
            @reset_parallel_kernel()
        end;
    end;
)) end == nothing || true;
