using Test
import ParallelStencil
using Enzyme
using ParallelStencil.ParallelKernel
import ParallelStencil.ParallelKernel.AD
import ParallelStencil.ParallelKernel: @reset_parallel_kernel, @is_initialized, SUPPORTED_PACKAGES, PKG_CUDA, PKG_AMDGPU, PKG_METAL, PKG_THREADS, PKG_POLYESTER, INDICES, ARRAYTYPES, FIELDTYPES, SCALARTYPES
import ParallelStencil.ParallelKernel: @require, @prettystring, @gorgeousstring, @isgpu, @iscpu, interpolate
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
@static if PKG_METAL in TEST_PACKAGES
    import Metal
    if !Metal.functional() TEST_PACKAGES = filter!(x->x≠PKG_METAL, TEST_PACKAGES) end
end
@static if PKG_POLYESTER in TEST_PACKAGES
    import Polyester
end
Base.retry_load_extensions() # Potentially needed to load the extensions after the packages have been filtered.

macro compute(A)              esc(:($(INDICES[1]) + ($(INDICES[2])-1)*size($A,1))) end
macro compute_with_aliases(A) esc(:(ix            + (iz           -1)*size($A,1))) end


@static for package in TEST_PACKAGES
    FloatDefault = (package == PKG_METAL) ? Float32 : Float64 # Metal does not support Float64

eval(:(
    @testset "$(basename(@__FILE__)) (package: $(nameof($package)))" begin
        @testset "1. parallel macros" begin
            @require !@is_initialized()
            @init_parallel_kernel($package, $FloatDefault)
            @require @is_initialized()
            @testset "@parallel" begin
                @static if $package == $PKG_CUDA
                    call = @prettystring(1, @parallel f(A))
                    @test occursin("CUDA.@cuda blocks = ParallelStencil.ParallelKernel.compute_nblocks(length.(ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A))), ParallelStencil.ParallelKernel.compute_nthreads(length.(ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A))); nthreads_x_max = 32)) threads = ParallelStencil.ParallelKernel.compute_nthreads(length.(ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A))); nthreads_x_max = 32) stream = CUDA.stream() f(A, ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A)), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A)))[1])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A)))[2])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A)))[3])))", call)
                    @test occursin("CUDA.synchronize(CUDA.stream(); blocking = true)", call)
                    call = @prettystring(1, @parallel ranges f(A))
                    @test occursin("CUDA.@cuda blocks = ParallelStencil.ParallelKernel.compute_nblocks(length.(ParallelStencil.ParallelKernel.promote_ranges(ranges)), ParallelStencil.ParallelKernel.compute_nthreads(length.(ParallelStencil.ParallelKernel.promote_ranges(ranges)); nthreads_x_max = 32)) threads = ParallelStencil.ParallelKernel.compute_nthreads(length.(ParallelStencil.ParallelKernel.promote_ranges(ranges)); nthreads_x_max = 32) stream = CUDA.stream() f(A, ParallelStencil.ParallelKernel.promote_ranges(ranges), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ranges))[1])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ranges))[2])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ranges))[3])))", call)
                    call = @prettystring(1, @parallel nblocks nthreads f(A))
                    @test occursin("CUDA.@cuda blocks = nblocks threads = nthreads stream = CUDA.stream() f(A, ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads)), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads)))[1])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads)))[2])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads)))[3])))", call)
                    call = @prettystring(1, @parallel ranges nblocks nthreads f(A))
                    @test occursin("CUDA.@cuda blocks = nblocks threads = nthreads stream = CUDA.stream() f(A, ParallelStencil.ParallelKernel.promote_ranges(ranges), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ranges))[1])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ranges))[2])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ranges))[3])))", call)
                    call = @prettystring(1, @parallel nblocks nthreads stream=mystream f(A))
                    @test occursin("CUDA.@cuda blocks = nblocks threads = nthreads stream = mystream f(A, ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads)), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads)))[1])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads)))[2])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads)))[3])))", call)
                elseif $package == $PKG_AMDGPU
                    call = @prettystring(1, @parallel f(A))
                    @test occursin("AMDGPU.@roc gridsize = ParallelStencil.ParallelKernel.compute_nblocks(length.(ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A))), ParallelStencil.ParallelKernel.compute_nthreads(length.(ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A))); nthreads_x_max = 64)) groupsize = ParallelStencil.ParallelKernel.compute_nthreads(length.(ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A))); nthreads_x_max = 64) stream = AMDGPU.stream() f(A, ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A)), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A)))[1])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A)))[2])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A)))[3])))", call)
                    @test occursin("AMDGPU.synchronize(AMDGPU.stream(); blocking = true)", call)
                    call = @prettystring(1, @parallel ranges f(A))
                    @test occursin("AMDGPU.@roc gridsize = ParallelStencil.ParallelKernel.compute_nblocks(length.(ParallelStencil.ParallelKernel.promote_ranges(ranges)), ParallelStencil.ParallelKernel.compute_nthreads(length.(ParallelStencil.ParallelKernel.promote_ranges(ranges)); nthreads_x_max = 64)) groupsize = ParallelStencil.ParallelKernel.compute_nthreads(length.(ParallelStencil.ParallelKernel.promote_ranges(ranges)); nthreads_x_max = 64) stream = AMDGPU.stream() f(A, ParallelStencil.ParallelKernel.promote_ranges(ranges), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ranges))[1])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ranges))[2])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ranges))[3])))", call)
                    call = @prettystring(1, @parallel nblocks nthreads f(A))
                    @test occursin("AMDGPU.@roc gridsize = nblocks groupsize = nthreads stream = AMDGPU.stream() f(A, ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads)), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads)))[1])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads)))[2])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads)))[3])))", call)
                    call = @prettystring(1, @parallel ranges nblocks nthreads f(A))
                    @test occursin("AMDGPU.@roc gridsize = nblocks groupsize = nthreads stream = AMDGPU.stream() f(A, ParallelStencil.ParallelKernel.promote_ranges(ranges), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ranges))[1])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ranges))[2])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ranges))[3])))", call)
                    call = @prettystring(1, @parallel nblocks nthreads stream=mystream f(A))
                    @test occursin("AMDGPU.@roc gridsize = nblocks groupsize = nthreads stream = mystream f(A, ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads)), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads)))[1])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads)))[2])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads)))[3])))", call)
                elseif $package == $PKG_METAL
                    call = @prettystring(1, @parallel f(A))
                    @test occursin("Metal.@metal groups = ParallelStencil.ParallelKernel.compute_nblocks(length.(ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A))), ParallelStencil.ParallelKernel.compute_nthreads(length.(ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A))); nthreads_x_max = 32)) threads = ParallelStencil.ParallelKernel.compute_nthreads(length.(ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A))); nthreads_x_max = 32) queue = Metal.global_queue(Metal.device()) f(A, ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A)), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A)))[1])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A)))[2])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.get_ranges(A)))[3])))", call)
                    @test occursin("Metal.synchronize(Metal.global_queue(Metal.device()))", call)
                    call = @prettystring(1, @parallel ranges f(A))
                    @test occursin("Metal.@metal groups = ParallelStencil.ParallelKernel.compute_nblocks(length.(ParallelStencil.ParallelKernel.promote_ranges(ranges)), ParallelStencil.ParallelKernel.compute_nthreads(length.(ParallelStencil.ParallelKernel.promote_ranges(ranges)); nthreads_x_max = 32)) threads = ParallelStencil.ParallelKernel.compute_nthreads(length.(ParallelStencil.ParallelKernel.promote_ranges(ranges)); nthreads_x_max = 32) queue = Metal.global_queue(Metal.device()) f(A, ParallelStencil.ParallelKernel.promote_ranges(ranges), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ranges))[1])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ranges))[2])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ranges))[3])))", call)
                    call = @prettystring(1, @parallel nblocks nthreads f(A))
                    @test occursin("Metal.@metal groups = nblocks threads = nthreads queue = Metal.global_queue(Metal.device()) f(A, ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads)), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads)))[1])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads)))[2])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads)))[3])))", call)
                    call = @prettystring(1, @parallel ranges nblocks nthreads f(A))
                    @test occursin("Metal.@metal groups = nblocks threads = nthreads queue = Metal.global_queue(Metal.device()) f(A, ParallelStencil.ParallelKernel.promote_ranges(ranges), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ranges))[1])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ranges))[2])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ranges))[3])))", call)
                    call = @prettystring(1, @parallel nblocks nthreads stream=mystream f(A))
                    @test occursin("Metal.@metal groups = nblocks threads = nthreads queue = mystream f(A, ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads)), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads)))[1])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads)))[2])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads)))[3])))", call)
                elseif @iscpu($package)
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
                        y::Float32
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
                @static if $package == $PKG_THREADS && VERSION.minor < 11 # TODO: remove restriction to Julia version < 1.11 once Enzyme support is available.
                    N = 16
                    a = 6.5
                    A = @rand(N)
                    B = @rand(N)
                    Ā = @ones(N)
                    B̄ = @ones(N)
                    A_ref = Array(A)
                    B_ref = Array(B)
                    Ā_ref = ones($FloatDefault, N)
                    B̄_ref = ones($FloatDefault, N)
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
                    @parallel configcall=f!(A, B, a) AD.autodiff_deferred!(Enzyme.Reverse, f!, Const, DuplicatedNoNeed(A, Ā), DuplicatedNoNeed(B, B̄), Const(a)) # NOTE: f! is automatically promoted to Const.
                    Enzyme.autodiff_deferred(Enzyme.Reverse, Const(g!), Const, DuplicatedNoNeed(A_ref, Ā_ref), DuplicatedNoNeed(B_ref, B̄_ref), Const(a))
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
                $(interpolate(:__T__, ARRAYTYPES, :(
                    @testset "Data.__T__ to Data.Device.__T__" begin
                        @static if @isgpu($package)
                            expansion = @prettystring(1, @parallel_indices (ix,iy) f(A::Data.__T__, B::Data.__T__, c::T) where T <: Integer = (A[ix,iy] = B[ix,iy]^c; return))
                            @test occursin("f(A::Data.Device.__T__, B::Data.Device.__T__,", expansion)
                        end
                    end
                )))
                $(interpolate(:__T__, FIELDTYPES, :(
                    @testset "Data.Fields.__T__ to Data.Fields.Device.__T__" begin
                        @static if @isgpu($package)
                            expansion = @prettystring(1, @parallel_indices (ix,iy) f(A::Data.Fields.__T__, B::Data.Fields.__T__, c::T) where T <: Integer = (A[ix,iy] = B[ix,iy]^c; return))
                            @test occursin("f(A::Data.Fields.Device.__T__, B::Data.Fields.Device.__T__,", expansion)
                        end
                    end
                )))
                # NOTE: the following GPU tests fail, because the Fields module cannot be imported.
                # @testset "Fields.Field to Data.Fields.Device.Field" begin
                #     @static if @isgpu($package)
                #             import .Data.Fields
                #             expansion = @prettystring(1, @parallel_indices (ix,iy) f(A::Fields.Field, B::Fields.Field, c::T) where T <: Integer = (A[ix,iy] = B[ix,iy]^c; return))
                #             @test occursin("f(A::Data.Fields.Device.Field, B::Data.Fields.Device.Field,", expansion)
                #     end
                # end
                # @testset "Field to Data.Fields.Device.Field" begin
                #     @static if @isgpu($package)
                #             using .Data.Fields
                #             expansion = @prettystring(1, @parallel_indices (ix,iy) f(A::Field, B::Field, c::T) where T <: Integer = (A[ix,iy] = B[ix,iy]^c; return))
                #             @test occursin("f(A::Data.Fields.Device.Field, B::Data.Fields.Device.Field,", expansion)
                #     end
                # end
                $(interpolate(:__T__, ARRAYTYPES, :(
                    @testset "TData.__T__ to TData.Device.__T__" begin
                        @static if @isgpu($package)
                            expansion = @prettystring(1, @parallel_indices (ix,iy) f(A::TData.__T__, B::TData.__T__, c::T) where T <: Integer = (A[ix,iy] = B[ix,iy]^c; return))
                            @test occursin("f(A::TData.Device.__T__, B::TData.Device.__T__,", expansion)
                        end
                    end
                )))
                $(interpolate(:__T__, FIELDTYPES, :(
                    @testset "TData.Fields.__T__ to TData.Fields.Device.__T__" begin
                        @static if @isgpu($package)
                            expansion = @prettystring(1, @parallel_indices (ix,iy) f(A::TData.Fields.__T__, B::TData.Fields.__T__, c::T) where T <: Integer = (A[ix,iy] = B[ix,iy]^c; return))
                            @test occursin("f(A::TData.Fields.Device.__T__, B::TData.Fields.Device.__T__,", expansion)
                        end
                    end
                )))
                # NOTE: the following GPU tests fail, because the Fields module cannot be imported.
                # @testset "Fields.Field to TData.Fields.Device.Field" begin
                #     @static if @isgpu($package)
                #             import .TData.Fields
                #             expansion = @prettystring(1, @parallel_indices (ix,iy) f(A::Fields.Field, B::Fields.Field, c::T) where T <: Integer = (A[ix,iy] = B[ix,iy]^c; return))
                #             @test occursin("f(A::TData.Fields.Device.Field, B::TData.Fields.Device.Field,", expansion)
                #     end
                # end
                # @testset "Field to TData.Fields.Device.Field" begin
                #     @static if @isgpu($package)
                    #             using .TData.Fields
                #             expansion = @prettystring(1, @parallel_indices (ix,iy) f(A::Field, B::Field, c::T) where T <: Integer = (A[ix,iy] = B[ix,iy]^c; return))
                #             @test occursin("f(A::TData.Fields.Device.Field, B::TData.Fields.Device.Field,", expansion)
                #     end
                # end
                $(interpolate(:__T__, ARRAYTYPES, :(
                    @testset "Nested Data.__T__ to Data.Device.__T__" begin
                        @static if @isgpu($package)
                            expansion = @prettystring(1, @parallel_indices (ix,iy) f(A::NamedTuple{T1, NTuple{T2,T3}} where {T1,T2} where T3 <: Data.__T__, c::T) where T <: Integer = (A[ix,iy] = B[ix,iy]^c; return))
                            @test occursin("f(A::((NamedTuple{T1, NTuple{T2, T3}} where {T1, T2}) where T3 <: Data.Device.__T__),", expansion)
                        end
                    end
                )))
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
                @static if $package != $PKG_POLYESTER
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
                end
            end;
            @testset "@parallel_async" begin
                @static if @isgpu($package)
                    call = @prettystring(1, @parallel_async f(A))
                    @test !occursin("synchronize", call)
                end;
            end;
            @testset "@synchronize" begin
                @static if $package == $PKG_CUDA
                    @test @prettystring(1, @synchronize()) == "CUDA.synchronize(; blocking = true)"
                    @test @prettystring(1, @synchronize(mystream)) == "CUDA.synchronize(mystream; blocking = true)"
                elseif $package == $PKG_AMDGPU
                    @test @prettystring(1, @synchronize()) == "AMDGPU.synchronize(; blocking = true)"
                    @test @prettystring(1, @synchronize(mystream)) == "AMDGPU.synchronize(mystream; blocking = true)"
                end;
            end;
            @reset_parallel_kernel()
        end;
        @testset "2. parallel macros (literal conversion)" begin
            if $package != $PKG_METAL
                @testset "@parallel_indices (Float64)" begin
                    @require !@is_initialized()
                    @init_parallel_kernel($package, Float64)
                    @require @is_initialized()
                    expansion = @gorgeousstring(@parallel_indices (ix) f!(A) = (A[ix] = A[ix] + 1.0; return))
                    @test occursin("A[ix] = A[ix] + 1.0\n", expansion)
                    @reset_parallel_kernel()
                end;
            end
            @testset "@parallel_indices (Float32)" begin
                @require !@is_initialized()
                @init_parallel_kernel($package, Float32)
                @require @is_initialized()
                expansion = @gorgeousstring(@parallel_indices (ix) f!(A) = (A[ix] = A[ix] + 1.0f0; return))
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
            if $package != $PKG_METAL
                @testset "@parallel_indices (ComplexF64)" begin
                    @require !@is_initialized()
                    @init_parallel_kernel($package, ComplexF64)
                    @require @is_initialized()
                    expansion = @gorgeousstring(@parallel_indices (ix) f!(A) = (A[ix] = 2.0f0 - 1.0f0im - A[ix] + 1.0f0; return))
                    @test occursin("A[ix] = ((2.0 - 1.0im) - A[ix]) + 1.0\n", expansion)
                    @reset_parallel_kernel()
                end;
            end
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
                @init_parallel_kernel($package, $FloatDefault, inbounds=true)
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
            $(interpolate(:__T__, ARRAYTYPES, :(
                @testset "Data.__T__{T2} to Data.Device.__T__{T2}" begin
                    @static if @isgpu($package)
                        expansion = @prettystring(1, @parallel_indices (ix,iy) f(A::Data.__T__{T2}, B::Data.__T__{T2}, c<:Integer) where T2 <: Union{Float32, Float64}  = (A[ix,iy] = B[ix,iy]^c; return))
                        @test occursin("f(A::Data.Device.__T__{T2}, B::Data.Device.__T__{T2},", expansion)
                    end
                end;
            )))
            $(interpolate(:__T__, FIELDTYPES, :(
                @testset "Data.Fields.__T__{T2} to Data.Fields.Device.__T__{T2}" begin
                    @static if @isgpu($package)
                        expansion = @prettystring(1, @parallel_indices (ix,iy) f(A::Data.Fields.__T__{T2}, B::Data.Fields.__T__{T2}, c<:Integer) where T2 <: Union{Float32, Float64}  = (A[ix,iy] = B[ix,iy]^c; return))
                        @test occursin("f(A::Data.Fields.Device.__T__{T2}, B::Data.Fields.Device.__T__{T2},", expansion)
                    end
                end;
            )))
            @reset_parallel_kernel()
        end;
        @testset "5. Exceptions" begin
            @require !@is_initialized()
            @init_parallel_kernel($package, $FloatDefault)
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
))

end == nothing || true;
