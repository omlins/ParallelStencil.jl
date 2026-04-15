using Test
using ParallelStencil
import ParallelStencil: @reset_parallel_stencil, @is_initialized, SUPPORTED_PACKAGES, PKG_CUDA, PKG_AMDGPU, PKG_METAL, PKG_THREADS, PKG_POLYESTER, PKG_KERNELABSTRACTIONS, @select_hardware, @current_hardware, INDICES, INDICES_INN, INDICES_DIR, ARRAYTYPES, FIELDTYPES, SCALARTYPES
import ParallelStencil: @require, @prettystring, @gorgeousstring, @isgpu, @iscpu, interpolate, @metadata
import ParallelStencil: checkargs_parallel, validate_body, parallel, parallel_indices
using ParallelStencil.Exceptions
using ParallelStencil.FiniteDifferences3D
using ParallelStencil.FieldAllocators
import ParallelStencil.FieldAllocators: @XXYYZField, @XYYZZField
import ParallelStencil.ParallelKernel: handle, @ka
ix, iy, iz = INDICES[1], INDICES[2], INDICES[3]
ixi, iyi, izi = INDICES_INN[1], INDICES_INN[2], INDICES_INN[3]
ixd, iyd, izd = INDICES_DIR[1], INDICES_DIR[2], INDICES_DIR[3]
ix_s, iy_s, iz_s = "var\"$ix\"", "var\"$iy\"", "var\"$iz\""
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
@static if PKG_KERNELABSTRACTIONS in TEST_PACKAGES
    import KernelAbstractions
    if !KernelAbstractions.functional(KernelAbstractions.CPU()) TEST_PACKAGES = filter!(x->x≠PKG_KERNELABSTRACTIONS, TEST_PACKAGES) end
    const KA_GPU_SYMBOLS = filter(!isnothing, (
        PKG_CUDA in TEST_PACKAGES ? :gpu_cuda : nothing,
        PKG_AMDGPU in TEST_PACKAGES ? :gpu_amd : nothing,
        PKG_METAL in TEST_PACKAGES ? :gpu_metal : nothing,
        (isdefined(ParallelStencil.ParallelKernel, :PKG_ONEAPI) && ParallelStencil.ParallelKernel.PKG_ONEAPI in TEST_PACKAGES) ? :gpu_oneapi : nothing
    ))
end
@static if PKG_POLYESTER in TEST_PACKAGES
    import Polyester
end
Base.retry_load_extensions() # Potentially needed to load the extensions after the packages have been filtered.

parallel_indices(args::Union{Symbol,Expr}...; package::Symbol=ParallelStencil.ParallelKernel.get_package(@__MODULE__)) = (ParallelStencil.checkargs_parallel_indices(args...); ParallelStencil.parallel_indices(LineNumberNode(@__LINE__, Symbol(@__FILE__)), @__MODULE__, args...; package=package))


@static for package in TEST_PACKAGES
    FloatDefault = (package == PKG_METAL) ? Float32 : Float64 # Metal does not support Float64

eval(:(
    @testset "$(basename(@__FILE__)) (package: $(nameof($package)))" begin
        @testset "1. parallel macros" begin
            @require !@is_initialized()
            @init_parallel_stencil($package, $FloatDefault, 3, nonconst_metadata=true)
            @require @is_initialized()
            @static if $package == $PKG_KERNELABSTRACTIONS
                @select_hardware(:cpu)
            end
            @testset "@parallel <kernelcall>" begin # NOTE: calls must go to ParallelStencil.ParallelKernel.parallel and must therefore give the same result as in ParallelKernel, except for memopt tests (tests copied 1-to-1 from there).
                @static if $package == $PKG_CUDA
                    @parallel_indices (ix, iy, iz) memopt=true loopsize=3 function f_memopt!(A2, A)
                        if (1 < iz < size(A2, 3))
                            A2[ix, iy, iz] = A[ix, iy, iz + 1] - 2 * A[ix, iy, iz] + A[ix, iy, iz - 1]
                        end
                        return
                    end
                    @parallel_indices (ix, iy) ndims=2 memopt=true loopdim=2 loopsize=3 function f_memopt_2d!(A2, A)
                        if (1 < iy < size(A2, 2))
                            A2[ix, iy] = A[ix, iy + 1] - 2 * A[ix, iy] + A[ix, iy - 1]
                        end
                        return
                    end
                    call = @prettystring(1, @parallel f(A))
                    @test occursin("CUDA.@cuda", call)
                    @test occursin("ParallelStencil.compute_parallel_ranges(Val", call)
                    @test occursin("nb_parallel_indices", call)
                    @test occursin("CUDA.synchronize(CUDA.stream(); blocking = true)", call)
                    call = @prettystring(1, @parallel ranges f(A))
                    @test occursin("CUDA.@cuda blocks = ParallelStencil.ParallelKernel.compute_nblocks(length.(ParallelStencil.ParallelKernel.promote_ranges(ranges)), ParallelStencil.ParallelKernel.compute_nthreads(length.(ParallelStencil.ParallelKernel.promote_ranges(ranges)); nthreads_x_max = 32)) threads = ParallelStencil.ParallelKernel.compute_nthreads(length.(ParallelStencil.ParallelKernel.promote_ranges(ranges)); nthreads_x_max = 32) stream = CUDA.stream() f(A, ParallelStencil.ParallelKernel.promote_ranges(ranges), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ranges))[1])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ranges))[2])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ranges))[3])))", call)
                    call = @prettystring(1, @parallel nblocks nthreads f(A))
                    @test occursin("CUDA.@cuda blocks = nblocks threads = nthreads stream = CUDA.stream() f(A, ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads)), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads)))[1])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads)))[2])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads)))[3])))", call)
                    call = @prettystring(1, @parallel ranges nblocks nthreads f(A))
                    @test occursin("CUDA.@cuda blocks = nblocks threads = nthreads stream = CUDA.stream() f(A, ParallelStencil.ParallelKernel.promote_ranges(ranges), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ranges))[1])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ranges))[2])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ranges))[3])))", call)
                    call = @prettystring(1, @parallel nblocks nthreads stream=mystream f(A))
                    @test occursin("CUDA.@cuda blocks = nblocks threads = nthreads stream = mystream f(A, ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads)), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads)))[1])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads)))[2])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads)))[3])))", call)
                    call = @prettystring(2, @parallel f_memopt!(A2, A))
                    @test occursin("CUDA.@cuda", call)
                    @test occursin(".memopt", call)
                    @test occursin("ParallelStencil.compute_memopt_nthreads_nblocks(Val", call)
                    @test occursin("ParallelStencil.compute_memopt_shmem(Val", call)
                    @test occursin("f_memopt!(A2, A, ParallelStencil.ParallelKernel.promote_ranges(", call)
                    call = @prettystring(2, @parallel ranges f_memopt!(A2, A))
                    @test occursin("ParallelStencil.compute_memopt_nthreads_nblocks(Val", call)
                    @test occursin("ParallelStencil.compute_memopt_shmem(Val", call)
                    @test occursin("CUDA.@cuda blocks = var\"##nblocks", call)
                    @test occursin("f_memopt!(A2, A, ParallelStencil.ParallelKernel.promote_ranges(ranges)", call)
                    call = @prettystring(2, @parallel f_memopt_2d!(A2, A))
                    @test occursin("CUDA.@cuda", call)
                    @test occursin(".memopt", call)
                    @test occursin("ParallelStencil.compute_memopt_nthreads_nblocks(Val", call)
                    @test occursin("ParallelStencil.compute_memopt_shmem(Val", call)
                    @test occursin("f_memopt_2d!(A2, A, ParallelStencil.ParallelKernel.promote_ranges(", call)
                elseif $package == $PKG_AMDGPU
                    @parallel_indices (ix, iy, iz) memopt=true loopsize=3 function f_memopt!(A2, A)
                        if (1 < iz < size(A2, 3))
                            A2[ix, iy, iz] = A[ix, iy, iz + 1] - 2 * A[ix, iy, iz] + A[ix, iy, iz - 1]
                        end
                        return
                    end
                    @parallel_indices (ix, iy) ndims=2 memopt=true loopdim=2 loopsize=3 function f_memopt_2d!(A2, A)
                        if (1 < iy < size(A2, 2))
                            A2[ix, iy] = A[ix, iy + 1] - 2 * A[ix, iy] + A[ix, iy - 1]
                        end
                        return
                    end
                    call = @prettystring(1, @parallel f(A))
                    @test occursin("AMDGPU.@roc", call)
                    @test occursin("ParallelStencil.compute_parallel_ranges(Val", call)
                    @test occursin("nb_parallel_indices", call)
                    @test occursin("AMDGPU.synchronize(AMDGPU.stream(); blocking = true)", call)
                    call = @prettystring(1, @parallel ranges f(A))
                    @test occursin("AMDGPU.@roc gridsize = ParallelStencil.ParallelKernel.compute_nblocks(length.(ParallelStencil.ParallelKernel.promote_ranges(ranges)), ParallelStencil.ParallelKernel.compute_nthreads(length.(ParallelStencil.ParallelKernel.promote_ranges(ranges)); nthreads_x_max = 64)) groupsize = ParallelStencil.ParallelKernel.compute_nthreads(length.(ParallelStencil.ParallelKernel.promote_ranges(ranges)); nthreads_x_max = 64) stream = AMDGPU.stream() f(A, ParallelStencil.ParallelKernel.promote_ranges(ranges), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ranges))[1])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ranges))[2])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ranges))[3])))", call)
                    call = @prettystring(1, @parallel nblocks nthreads f(A))
                    @test occursin("AMDGPU.@roc gridsize = nblocks groupsize = nthreads stream = AMDGPU.stream() f(A, ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads)), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads)))[1])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads)))[2])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads)))[3])))", call)
                    call = @prettystring(1, @parallel ranges nblocks nthreads f(A))
                    @test occursin("AMDGPU.@roc gridsize = nblocks groupsize = nthreads stream = AMDGPU.stream() f(A, ParallelStencil.ParallelKernel.promote_ranges(ranges), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ranges))[1])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ranges))[2])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ranges))[3])))", call)
                    call = @prettystring(1, @parallel nblocks nthreads stream=mystream f(A))
                    @test occursin("AMDGPU.@roc gridsize = nblocks groupsize = nthreads stream = mystream f(A, ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads)), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads)))[1])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads)))[2])), (Int64)(length((ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads)))[3])))", call)
                    call = @prettystring(2, @parallel f_memopt!(A2, A))
                    @test occursin("AMDGPU.@roc", call)
                    @test occursin(".memopt", call)
                    @test occursin("ParallelStencil.compute_memopt_nthreads_nblocks(Val", call)
                    @test occursin("ParallelStencil.compute_memopt_shmem(Val", call)
                    @test occursin("f_memopt!(A2, A, ParallelStencil.ParallelKernel.promote_ranges(", call)
                    call = @prettystring(2, @parallel ranges f_memopt!(A2, A))
                    @test occursin("ParallelStencil.compute_memopt_nthreads_nblocks(Val", call)
                    @test occursin("ParallelStencil.compute_memopt_shmem(Val", call)
                    @test occursin("AMDGPU.@roc gridsize = var\"##nblocks", call)
                    @test occursin("f_memopt!(A2, A, ParallelStencil.ParallelKernel.promote_ranges(ranges)", call)
                    call = @prettystring(2, @parallel f_memopt_2d!(A2, A))
                    @test occursin("AMDGPU.@roc", call)
                    @test occursin(".memopt", call)
                    @test occursin("ParallelStencil.compute_memopt_nthreads_nblocks(Val", call)
                    @test occursin("ParallelStencil.compute_memopt_shmem(Val", call)
                    @test occursin("f_memopt_2d!(A2, A, ParallelStencil.ParallelKernel.promote_ranges(", call)
                elseif $package == $PKG_KERNELABSTRACTIONS
                    call = @prettystring(1, @parallel f(A))
                    @test occursin("ParallelStencil.ParallelKernel.@ka", call)
                    @test occursin("handle(ParallelStencil.ParallelKernel.current_hardware(@__MODULE__()), :$PKG_KERNELABSTRACTIONS)", call)
                    @test occursin("compute_nblocks", call)
                    @test occursin("compute_nthreads", call)
                    @test occursin("ParallelStencil.compute_parallel_ranges(Val", call)
                    @test occursin("nb_parallel_indices", call)
                    @test !occursin("CUDA.@cuda", call)
                    @test !occursin("AMDGPU.@roc", call)
                    call = @prettystring(1, @parallel ranges f(A))
                    @test occursin("ParallelStencil.ParallelKernel.@ka", call)
                    @test occursin("handle(ParallelStencil.ParallelKernel.current_hardware(@__MODULE__()), :$PKG_KERNELABSTRACTIONS)", call)
                    call = @prettystring(1, @parallel nblocks nthreads f(A))
                    @test occursin("ParallelStencil.ParallelKernel.@ka", call)
                    @test occursin("handle(ParallelStencil.ParallelKernel.current_hardware(@__MODULE__()), :$PKG_KERNELABSTRACTIONS)", call)
                    call = @prettystring(1, @parallel ranges nblocks nthreads f(A))
                    @test occursin("ParallelStencil.ParallelKernel.@ka", call)
                    @test occursin("handle(ParallelStencil.ParallelKernel.current_hardware(@__MODULE__()), :$PKG_KERNELABSTRACTIONS)", call)
                    call = @prettystring(1, @parallel nblocks nthreads stream=mystream f(A))
                    @test occursin("ParallelStencil.ParallelKernel.@ka", call)
                    @test occursin("queue = mystream", call)
                    # call = @prettystring(2, @parallel memopt=true f(A))
                    # @test occursin("ParallelStencil.ParallelKernel.@ka", call)
                    # call = @prettystring(2, @parallel ranges memopt=true f(A))
                    # @test occursin("ParallelStencil.ParallelKernel.@ka", call)
                elseif @iscpu($package)
                    call = @prettystring(1, @parallel f(A))
                    @test occursin("f(A, ParallelStencil.ParallelKernel.promote_ranges(", call)
                    @test occursin("ParallelStencil.compute_parallel_ranges(Val", call)
                    @test occursin("nb_parallel_indices", call)
                    call = @prettystring(1, @parallel ranges f(A))
                    @test occursin(".memopt", call)
                    @test occursin("f(A, ParallelStencil.ParallelKernel.promote_ranges(ranges)", call)
                    call = @prettystring(1, @parallel nblocks nthreads f(A))
                    @test occursin(".memopt", call)
                    @test occursin("f(A, ParallelStencil.ParallelKernel.promote_ranges(ParallelStencil.ParallelKernel.compute_ranges(nblocks .* nthreads))", call)
                    call = @prettystring(1, @parallel ranges nblocks nthreads f(A))
                    @test occursin(".memopt", call)
                    @test occursin("f(A, ParallelStencil.ParallelKernel.promote_ranges(ranges)", call)
                    call = @prettystring(1, @parallel stream=mystream f(A))
                    @test occursin("f(A, ParallelStencil.ParallelKernel.promote_ranges(", call)
                    @test occursin("ParallelStencil.compute_parallel_ranges(Val", call)
                    @test occursin("nb_parallel_indices", call)
                    @parallel_indices (ix, iy, iz) memopt=true loopsize=3 function f_memopt!(A2, A)
                        if (1 < iz < size(A2, 3))
                            A2[ix, iy, iz] = A[ix, iy, iz + 1] - 2 * A[ix, iy, iz] + A[ix, iy, iz - 1]
                        end
                        return
                    end
                    @parallel_indices (ix, iy) ndims=2 memopt=true loopdim=2 loopsize=3 function f_memopt_2d!(A2, A)
                        if (1 < iy < size(A2, 2))
                            A2[ix, iy] = A[ix, iy + 1] - 2 * A[ix, iy] + A[ix, iy - 1]
                        end
                        return
                    end
                    call = @prettystring(2, @parallel f_memopt!(A2, A))
                    @test occursin(".memopt", call)
                    @test occursin("ParallelStencil.compute_memopt_nthreads_nblocks(Val", call)
                    @test occursin("ParallelStencil.compute_memopt_shmem(Val", call)
                    @test occursin("f_memopt!(A2, A, ParallelStencil.ParallelKernel.promote_ranges(", call)
                    call = @prettystring(2, @parallel ranges f_memopt!(A2, A))
                    @test occursin("ParallelStencil.compute_memopt_nthreads_nblocks(Val", call)
                    @test occursin("ParallelStencil.compute_memopt_shmem(Val", call)
                    @test occursin("f_memopt!(A2, A, ParallelStencil.ParallelKernel.promote_ranges(ranges)", call)
                    call = @prettystring(2, @parallel f_memopt_2d!(A2, A))
                    @test occursin(".memopt", call)
                    @test occursin("ParallelStencil.compute_memopt_nthreads_nblocks(Val", call)
                    @test occursin("ParallelStencil.compute_memopt_shmem(Val", call)
                    @test occursin("f_memopt_2d!(A2, A, ParallelStencil.ParallelKernel.promote_ranges(", call)
                end;
            end;
            @testset "KernelAbstractions runtime reselection" begin
                @static if $package == $PKG_KERNELABSTRACTIONS
                    @require KernelAbstractions.functional(KernelAbstractions.CPU())
                    N = 8
                    @parallel_indices (ix) function ka_double!(A)
                        A[ix] = A[ix] + one(eltype(A))
                        return
                    end
                    valid_symbols = (:cpu, KA_GPU_SYMBOLS...)
                    last_symbol = :hw_none
                    for symbol in (:cpu, KA_GPU_SYMBOLS...)
                        if symbol == :gpu_cuda && !(PKG_CUDA in TEST_PACKAGES)
                            @test_skip "KernelAbstractions GPU symbol :gpu_cuda unavailable"
                            continue
                        elseif symbol == :gpu_amd && !(PKG_AMDGPU in TEST_PACKAGES)
                            @test_skip "KernelAbstractions GPU symbol :gpu_amd unavailable"
                            continue
                        elseif symbol == :gpu_metal && !(PKG_METAL in TEST_PACKAGES)
                            @test_skip "KernelAbstractions GPU symbol :gpu_metal unavailable"
                            continue
                        elseif symbol == :gpu_oneapi && !(isdefined(ParallelStencil.ParallelKernel, :PKG_ONEAPI) && ParallelStencil.ParallelKernel.PKG_ONEAPI in TEST_PACKAGES)
                            @test_skip "KernelAbstractions GPU symbol :gpu_oneapi unavailable"
                            continue
                        end
                        @select_hardware(symbol)
                        A = @zeros(N)
                        @parallel (1:N) ka_double!(A)
                        @test all(Array(A) .== one(eltype(A)))
                        last_symbol = symbol
                    end
                    @test @current_hardware() in valid_symbols
                    @test @current_hardware() == last_symbol
                    @select_hardware(:cpu)
                    @test @current_hardware() == :cpu
                end
            end
            @static if $package != $PKG_POLYESTER # Enzyme does not support Polyester.
              @testset "@parallel ∇" begin
                  @test @prettystring(1, @parallel ∇=B->B̄ f!(A, B, a)) == "@parallel configcall = f!(A, B, a) ParallelStencil.ParallelKernel.AD.autodiff_deferred!(Enzyme.Reverse, f!, Enzyme.Const(A), Enzyme.DuplicatedNoNeed(B, B̄), Enzyme.Const(a))"
                  @test @prettystring(1, @parallel ∇=(A->Ā, B->B̄) f!(A, B, a)) == "@parallel configcall = f!(A, B, a) ParallelStencil.ParallelKernel.AD.autodiff_deferred!(Enzyme.Reverse, f!, Enzyme.DuplicatedNoNeed(A, Ā), Enzyme.DuplicatedNoNeed(B, B̄), Enzyme.Const(a))"
                  @test @prettystring(1, @parallel ∇=(A->Ā, B->B̄) ad_mode=Enzyme.Forward f!(A, B, a)) == "@parallel configcall = f!(A, B, a) ParallelStencil.ParallelKernel.AD.autodiff_deferred!(Enzyme.Forward, f!, Enzyme.DuplicatedNoNeed(A, Ā), Enzyme.DuplicatedNoNeed(B, B̄), Enzyme.Const(a))"
                  @test @prettystring(1, @parallel ∇=(A->Ā, B->B̄) ad_mode=Enzyme.Forward ad_annotations=(Duplicated=B) f!(A, B, a)) == "@parallel configcall = f!(A, B, a) ParallelStencil.ParallelKernel.AD.autodiff_deferred!(Enzyme.Forward, f!, Enzyme.DuplicatedNoNeed(A, Ā), Enzyme.Duplicated(B, B̄), Enzyme.Const(a))"
                  @test @prettystring(1, @parallel ∇=(A->Ā, B->B̄) ad_mode=Enzyme.Forward ad_annotations=(Duplicated=(B,A), Active=b) f!(A, B, a, b)) == "@parallel configcall = f!(A, B, a, b) ParallelStencil.ParallelKernel.AD.autodiff_deferred!(Enzyme.Forward, f!, Enzyme.Duplicated(A, Ā), Enzyme.Duplicated(B, B̄), Enzyme.Const(a), Enzyme.Active(b))"
                  @test @prettystring(1, @parallel ∇=(V.x->V̄.x, V.y->V̄.y) f!(V.x, V.y, a)) == "@parallel configcall = f!(V.x, V.y, a) ParallelStencil.ParallelKernel.AD.autodiff_deferred!(Enzyme.Reverse, f!, Enzyme.DuplicatedNoNeed(V.x, V̄.x), Enzyme.DuplicatedNoNeed(V.y, V̄.y), Enzyme.Const(a))"
                  @static if $package == $PKG_KERNELABSTRACTIONS
                      call = @prettystring(2, @parallel ∇=B->B̄ f!(A, B, a))
                      @test occursin("fname = f!", call)
                  end
                end;
            end;
            @testset "@parallel <kernel>" begin
                @testset "N substitution | ndims tuple expansion" begin
                    @testset "N substitution (N=3)" begin
                        expansion = @prettystring(1, @parallel N=3 f(A::Data.Array{N}, B::Data.Array{N}, c::Integer) where T <: PSNumber = (@all(A) = @all(B)^c; return))
                        @test occursin("f(A::Data.Array{3}, B::Data.Array{3},", expansion)
                    end;
                    @testset "N substitution (N=2)" begin
                        expansion = @prettystring(1, @parallel N=2 f(A::Data.Array{N}, B::Data.Array{N}, c::Integer) where T <: PSNumber = (@all(A) = @all(B)^c; return))
                        @test occursin("f(A::Data.Array{2}, B::Data.Array{2},", expansion)
                    end;
                    @testset "N substitution (N=1)" begin
                        expansion = @prettystring(1, @parallel N=1 f(A::Data.Array{N}, B::Data.Array{N}, c::Integer) where T <: PSNumber = (@all(A) = @all(B)^c; return))
                        @test occursin("f(A::Data.Array{1}, B::Data.Array{1},", expansion)
                    end;
                    @testset "N substitution (N=ndims)" begin
                        expansion = @prettystring(1, @parallel N=ndims f(A::Data.Array{N}, B::Data.Array{N}, c::Integer) where T <: PSNumber = (@all(A) = @all(B)^c; return))
                        @test occursin("f(A::Data.Array{3}, B::Data.Array{3},", expansion)
                    end;
                    @testset "N substitution (ndims=2, N=ndims+2)" begin
                        expansion = @prettystring(1, @parallel ndims=2 N=ndims+2 f(A::Data.Array{N}, B::Data.Array{N}, c::Integer) where T <: PSNumber = (@all(A) = @all(B)^c; return))
                        @test occursin("f(A::Data.Array{4}, B::Data.Array{4},", expansion)
                    end;
                    @testset "ndims tuple expansion (ndims=(1,2,3), N=ndims)" begin
                        expansion = @prettystring(2, @parallel ndims=(1,2,3) N=ndims f(A::Data.Array{T,N}, B::Data.Array{T,N}, c::Integer) where T <: PSNumber = (@all(A) = @all(B)^c; return))
                        @test occursin("@parallel ndims = 1 function f(A::Data.Array{T, 1}, B::Data.Array{T, 1},", expansion)
                        @test occursin("@parallel ndims = 2 function f(A::Data.Array{T, 2}, B::Data.Array{T, 2},", expansion)
                        @test occursin("@parallel ndims = 3 function f(A::Data.Array{T, 3}, B::Data.Array{T, 3},", expansion)
                    end;
                    @testset "ndims tuple expansion (ndims=(1,3), N=ndims.+1)" begin
                        expansion = @prettystring(2, @parallel ndims=(1,3) N=ndims.+1 f(A::Data.Array{T,N}, B::Data.Array{T,N}, c::Integer) where T <: PSNumber = (@all(A) = @all(B)^c; return))
                        @test occursin("@parallel ndims = 1 function f(A::Data.Array{T, 2}, B::Data.Array{T, 2},", expansion)
                        @test occursin("@parallel ndims = 3 function f(A::Data.Array{T, 4}, B::Data.Array{T, 4},", expansion)
                        @test !occursin("@parallel ndims = 2", expansion)
                    end;
                end;
                @testset "inbounds" begin
                    expansion = @prettystring(1, @parallel_indices (ix) inbounds=true f(A) = (2*A; return))
                    @test occursin("Base.@inbounds begin", expansion)
                    expansion = @prettystring(1, @parallel_indices (ix) inbounds=false f(A) = (2*A; return))
                    @test !occursin("Base.@inbounds begin", expansion)
                    expansion = @prettystring(1, @parallel_indices (ix) f(A) = (2*A; return))
                    @test !occursin("Base.@inbounds begin", expansion)
                end
                @testset "addition of range arguments" begin
                    expansion = @gorgeousstring(1, @parallel f(A, B, c::T) where T <: Integer = (@all(A) = @all(B)^c; return))
                    @test occursin("f(A, B, c::T, ranges::Tuple{UnitRange, UnitRange, UnitRange}, rangelength_x::Int64, rangelength_y::Int64, rangelength_z::Int64", expansion)
                end
                @static if $package != $PKG_KERNELABSTRACTIONS
                    @testset "memopt (2D)" begin
                        expansion = @gorgeousstring @parallel_indices (ix, iy) ndims=2 memopt=true loopdim=2 loopsize=3 function d2_memopt_decl_indices!(A2, A)
                            if (1 < iy < size(A2, 2))
                                A2[ix, iy] = A[ix, iy + 1] - 2 * A[ix, iy] + A[ix, iy - 1]
                            end
                            return
                        end
                        @test occursin("function d2_memopt_decl_indices!(A2, A", expansion)
                        @test occursin("A2[ix, iy]", expansion)
                        @test occursin("A[ix, iy + 1]", expansion)
                        @test !occursin("A2[ix, iy, iz]", expansion)
                        @test !occursin("useshmemhalos", expansion)
                        expansion = @gorgeousstring @parallel ndims=2 memopt=true optvars=B loopdim=2 loopsize=3 function d2_memopt_decl_kernel!(A2, B)
                            ParallelStencil.FiniteDifferences2D.@inn(A2) = ParallelStencil.FiniteDifferences2D.@d2_yi(B)
                            return
                        end
                        @test occursin("function d2_memopt_decl_kernel!(A2, B", expansion)
                        @test occursin("A2[ix + 1, iy + 1]", expansion)
                        @test occursin("B[ix + 1, iy + 2]", expansion)
                        @test !occursin("A2[ix + 1, iy + 1, iz + 1]", expansion)
                        @test !occursin("useshmemhalos", expansion)
                    end;
                end
                # NOTE: KernelAbstractions intentionally leaves convenience modules like Data.Device undefined; these checks only validate macro expansion shapes.
                $(interpolate(:__T__, ARRAYTYPES, :(
                    @testset "Data.__T__ to Data.Device.__T__" begin
                        @static if @isgpu($package)
                            expansion = @prettystring(1, @parallel f(A::Data.__T__, B::Data.__T__, c::T) where T <: Integer = (@all(A) = @all(B)^c; return))
                            @test occursin("f(A::Data.Device.__T__, B::Data.Device.__T__,", expansion)
                        end
                    end
                )))
                $(interpolate(:__T__, FIELDTYPES, :(
                    @testset "Data.Fields.__T__ to Data.Fields.Device.__T__" begin
                        @static if @isgpu($package)
                            expansion = @prettystring(1, @parallel f(A::Data.Fields.__T__, B::Data.Fields.__T__, c::T) where T <: Integer = (@all(A) = @all(B)^c; return))
                            @test occursin("f(A::Data.Fields.Device.__T__, B::Data.Fields.Device.__T__,", expansion)
                        end
                    end
                )))
                # NOTE: the following GPU tests fail, because the Fields module cannot be imported.
                # @testset "Fields.Field to Data.Fields.Device.Field" begin
                #     @static if @isgpu($package)
                #             import .Data.Fields
                #             expansion = @prettystring(1, @parallel f(A::Fields.Field, B::Fields.Field, c::T) where T <: Integer = (@all(A) = @all(B)^c; return))
                #             @test occursin("f(A::Data.Fields.Device.Field, B::Data.Fields.Device.Field,", expansion)
                #     end
                # end
                # @testset "Field to Data.Fields.Device.Field" begin
                #     @static if @isgpu($package)
                #             using .Data.Fields
                #             expansion = @prettystring(1, @parallel f(A::Field, B::Field, c::T) where T <: Integer = (@all(A) = @all(B)^c; return))
                #             @test occursin("f(A::Data.Fields.Device.Field, B::Data.Fields.Device.Field,", expansion)
                #     end
                # end
                $(interpolate(:__T__, ARRAYTYPES, :(
                    @testset "TData.__T__ to TData.Device.__T__" begin
                        @static if @isgpu($package)
                            expansion = @prettystring(1, @parallel f(A::TData.__T__, B::TData.__T__, c::T) where T <: Integer = (@all(A) = @all(B)^c; return))
                            @test occursin("f(A::TData.Device.__T__, B::TData.Device.__T__,", expansion)
                        end
                    end
                )))
                $(interpolate(:__T__, FIELDTYPES, :(
                    @testset "TData.Fields.__T__ to TData.Fields.Device.__T__" begin
                        @static if @isgpu($package)
                            expansion = @prettystring(1, @parallel f(A::TData.Fields.__T__, B::TData.Fields.__T__, c::T) where T <: Integer = (@all(A) = @all(B)^c; return))
                            @test occursin("f(A::TData.Fields.Device.__T__, B::TData.Fields.Device.__T__,", expansion)
                        end
                    end
                )))
                # NOTE: the following GPU tests fail, because the Fields module cannot be imported.
                # @testset "Fields.Field to TData.Fields.Device.Field" begin
                #     @static if @isgpu($package)
                #             import .TData.Fields
                #             expansion = @prettystring(1, @parallel f(A::Fields.Field, B::Fields.Field, c::T) where T <: Integer = (@all(A) = @all(B)^c; return))
                #             @test occursin("f(A::TData.Fields.Device.Field, B::TData.Fields.Device.Field,", expansion)
                #     end
                # end
                # @testset "Field to TData.Fields.Device.Field" begin
                #     @static if @isgpu($package)
                #             using .TData.Fields
                #             expansion = @prettystring(1, @parallel f(A::Field, B::Field, c::T) where T <: Integer = (@all(A) = @all(B)^c; return))
                #             @test occursin("f(A::TData.Fields.Device.Field, B::TData.Fields.Device.Field,", expansion)
                #     end
                # end
                @testset "@parallel <kernel> (3D)" begin
                    A  = @zeros(4, 5, 6)
                    @parallel function write_indices!(A)
                        @all(A) = $ix + ($iy-1)*size(A,1) + ($iz-1)*size(A,1)*size(A,2); # NOTE: $ix, $iy, $iz come from ParallelStencil.INDICES.
                        return
                    end
                    @parallel write_indices!(A);
                    @test all(Array(A) .== [ix + (iy-1)*size(A,1) + (iz-1)*size(A,1)*size(A,2) for ix=1:size(A,1), iy=1:size(A,2), iz=1:size(A,3)])
                end
                @testset "@parallel <kernel> (3D; on-the-fly)" begin
                    nx, ny, nz = 32, 8, 8
                    lam=dt=_dx=_dy=_dz = $FloatDefault(1)
                    T      = @zeros(nx, ny, nz);
                    T2     = @zeros(nx, ny, nz);
                    T2_ref = @zeros(nx, ny, nz);
                    Ci     = @ones(nx, ny, nz);
                    copy!(T, [ix + (iy-1)*size(T,1) + (iz-1)*size(T,1)*size(T,2) for ix=1:size(T,1), iy=1:size(T,2), iz=1:size(T,3)].^3);
                    @parallel function diffusion3D_step!(T2, T, Ci, lam::Data.Number, dt::$FloatDefault, _dx, _dy, _dz)
                        @all(qx)   = -lam*@d_xi(T)*_dx                                          # Fourier's law of heat conduction
                        @all(qy)   = -lam*@d_yi(T)*_dy                                          # ...
                        @all(qz)   = -lam*@d_zi(T)*_dz                                          # ...
                        @all(dTdt) = @inn(Ci)*(-@d_xa(qx)*_dx - @d_ya(qy)*_dy - @d_za(qz)*_dz)  # Conservation of energy
                        @inn(T2)   = @inn(T) + dt*@all(dTdt)                                    # Update of temperature
                        return
                    end
                    @parallel diffusion3D_step!(T2, T, Ci, lam, dt, _dx, _dy, _dz);
                    T2_ref[2:end-1,2:end-1,2:end-1] .= T[2:end-1,2:end-1,2:end-1] .+ dt.*(lam.*Ci[2:end-1,2:end-1,2:end-1].*(
                                              ((T[3:end  ,2:end-1,2:end-1] .- T[2:end-1,2:end-1,2:end-1]) .- (T[2:end-1,2:end-1,2:end-1] .- T[1:end-2,2:end-1,2:end-1])).*_dx^2
                                            + ((T[2:end-1,3:end  ,2:end-1] .- T[2:end-1,2:end-1,2:end-1]) .- (T[2:end-1,2:end-1,2:end-1] .- T[2:end-1,1:end-2,2:end-1])).*_dy^2
                                            + ((T[2:end-1,2:end-1,3:end  ] .- T[2:end-1,2:end-1,2:end-1]) .- (T[2:end-1,2:end-1,2:end-1] .- T[2:end-1,2:end-1,1:end-2])).*_dz^2)
                                            );
                    @test all(Array(T2) .== Array(T2_ref))
                end
                @static if $package in [$PKG_CUDA, $PKG_AMDGPU] # TODO add support for Metal
                    $(interpolate(:__padding__, (false, package!=PKG_POLYESTER), :( #TODO: this needs to be restored to (false, true) when Polyester supports padding.
                        @testset "(padding=$__padding__)" begin
                            @testset "@parallel memopt <kernel> (nx, ny, nz = x .* threads)" begin # NOTE: the following does not work for some reason: (nx, ny, nz = ($nx, $ny, $nz))" for (nx, ny, nz) in ((32, 8, 9), (32, 8, 8), (31, 7, 9), (33, 9, 9), (33, 7, 8))
                                nxyz = (32, 8, 8)
                                # threads      = (8, 4, 1)
                                # blocks       = ceil.(Int, (nx/threads[1], ny/threads[2], nz/LOOPSIZE))
                                # shmem        = (threads[1]+2)*(threads[2]+2)*sizeof(Float64)
                                @testset "@parallel_indices <kernel> (3D, memopt, stencilranges=0:0)" begin
                                    A  = @Field(nxyz);
                                    A2 = @Field(nxyz);
                                    copy!(A, [ix + (iy-1)*size(A,1) + (iz-1)*size(A,1)*size(A,2) for ix=1:size(A,1), iy=1:size(A,2), iz=1:size(A,3)].^3);
                                    @parallel_indices (ix,iy,iz) memopt=true loopsize=3 optvars=A optranges=(A=(0:0,0:0,0:0),) function copy_memopt!(A2, A)
                                        A2[ix,iy,iz] = A[ix,iy,iz]
                                        return
                                    end
                                    @parallel memopt=true copy_memopt!(A2, A);
                                    @test all(Array(A2) .== Array(A))
                                end
                                @testset "@parallel <kernel> (3D, memopt, stencilranges=0:0)" begin
                                    A  = @Field(nxyz);
                                    A2 = @Field(nxyz);
                                    copy!(A, [ix + (iy-1)*size(A,1) + (iz-1)*size(A,1)*size(A,2) for ix=1:size(A,1), iy=1:size(A,2), iz=1:size(A,3)].^3);
                                    @parallel memopt=true loopsize=3 optvars=A optranges=(A=(0:0,0:0,0:0),) function copy_memopt!(A2, A)
                                        @all(A2) = @all(A)
                                        return
                                    end
                                    @parallel memopt=true copy_memopt!(A2, A);
                                    @test all(Array(A2) .== Array(A))
                                end
                                @testset "@parallel_indices <kernel> (3D, memopt, stencilranges=(0:0, 0:0, -1:1); z-stencil)" begin
                                    A      = @Field(nxyz);
                                    A2     = @Field(nxyz);
                                    A2_ref = @Field(nxyz);
                                    copy!(A, [ix + (iy-1)*size(A,1) + (iz-1)*size(A,1)*size(A,2) for ix=1:size(A,1), iy=1:size(A,2), iz=1:size(A,3)].^3);
                                    @parallel_indices (ix,iy,iz) memopt=true loopsize=3 function d2_memopt!(A2, A)
                                        if (iz>1 && iz<size(A2,3))
                                            A2[ix,iy,iz] = A[ix,iy,iz+1] - 2*A[ix,iy,iz] + A[ix,iy,iz-1]
                                        end
                                        return
                                    end
                                    @parallel memopt=true d2_memopt!(A2, A);
                                    A2_ref[:,:,2:end-1] .= A[:,:,3:end] .- 2*A[:,:,2:end-1] .+ A[:,:,1:end-2];
                                    @test all(Array(A2) .== Array(A2_ref))
                                end
                                @testset "@parallel_indices <kernel> (3D, memopt, stencilranges=(0:0, -1:1, 0:0); y-stencil)" begin
                                    A      = @Field(nxyz);
                                    A2     = @Field(nxyz);
                                    A2_ref = @Field(nxyz);
                                    copy!(A, [ix + (iy-1)*size(A,1) + (iz-1)*size(A,1)*size(A,2) for ix=1:size(A,1), iy=1:size(A,2), iz=1:size(A,3)].^3);
                                    @parallel_indices (ix,iy,iz) memopt=true function d2_memopt!(A2, A)
                                        if (iy>1 && iy<size(A2,2))
                                            A2[ix,iy,iz] = A[ix,iy+1,iz] - 2*A[ix,iy,iz] + A[ix,iy-1,iz]
                                        end
                                        return
                                    end
                                    @parallel memopt=true d2_memopt!(A2, A);
                                    A2_ref[:,2:end-1,:] .= A[:,3:end,:] .- 2*A[:,2:end-1,:] .+ A[:,1:end-2,:];
                                    @test all(Array(A2) .== Array(A2_ref))
                                end
                                @testset "@parallel <kernel> (3D, memopt, stencilranges=(1:1, 1:1, 0:2); z-stencil)" begin
                                    A      = @Field(nxyz);
                                    A2     = @Field(nxyz);
                                    A2_ref = @Field(nxyz);
                                    copy!(A, [ix + (iy-1)*size(A,1) + (iz-1)*size(A,1)*size(A,2) for ix=1:size(A,1), iy=1:size(A,2), iz=1:size(A,3)].^3);
                                    @parallel memopt=true loopsize=3 function d2_memopt!(A2, A)
                                        @inn(A2) = @d2_zi(A)
                                        return
                                    end
                                    @parallel memopt=true d2_memopt!(A2, A);
                                    A2_ref[2:end-1,2:end-1,2:end-1] .= (A[2:end-1,2:end-1,3:end] .- A[2:end-1,2:end-1,2:end-1]) .- (A[2:end-1,2:end-1,2:end-1] .- A[2:end-1,2:end-1,1:end-2]);
                                    @test all(Array(A2) .== Array(A2_ref))
                                end
                                @testset "@parallel <kernel> (3D, memopt, stencilranges=(1:1, 0:2, 1:1); y-stencil)" begin
                                    A      = @Field(nxyz);
                                    A2     = @Field(nxyz);
                                    A2_ref = @Field(nxyz);
                                    copy!(A, [ix + (iy-1)*size(A,1) + (iz-1)*size(A,1)*size(A,2) for ix=1:size(A,1), iy=1:size(A,2), iz=1:size(A,3)].^3);
                                    @parallel memopt=true loopsize=3 function d2_memopt!(A2, A)
                                        @inn(A2) = @d2_yi(A)
                                        return
                                    end
                                    @parallel memopt=true d2_memopt!(A2, A);
                                    A2_ref[2:end-1,2:end-1,2:end-1] .= (A[2:end-1,3:end,2:end-1] .- A[2:end-1,2:end-1,2:end-1]) .- (A[2:end-1,2:end-1,2:end-1] .- A[2:end-1,1:end-2,2:end-1]);
                                    @test all(Array(A2) .== Array(A2_ref))
                                end
                                @testset "@parallel_indices <kernel> (3D, memopt, stencilranges=-1:1)" begin
                                    lam=dt=_dx=_dy=_dz = $FloatDefault(1)
                                    T      = @Field(nxyz);
                                    T2     = @Field(nxyz);
                                    T2_ref = @Field(nxyz);
                                    Ci     = @Field(nxyz, @ones);
                                    copy!(T, [ix + (iy-1)*size(T,1) + (iz-1)*size(T,1)*size(T,2) for ix=1:size(T,1), iy=1:size(T,2), iz=1:size(T,3)].^3);
                                    @parallel_indices (ix,iy,iz) memopt=true loopsize=3 function diffusion3D_step!(T2, T, Ci, lam, dt, _dx, _dy, _dz)
                                        if (1<ix<size(T2,1) && 1<iy<size(T2,2) && 1<iz<size(T2,3))
                                            T2[ix,iy,iz] = T[ix,iy,iz] + dt*(Ci[ix,iy,iz]*(
                                                            - ((-lam*(T[ix+1,iy,iz] - T[ix,iy,iz])*_dx) - (-lam*(T[ix,iy,iz] - T[ix-1,iy,iz])*_dx))*_dx
                                                            - ((-lam*(T[ix,iy+1,iz] - T[ix,iy,iz])*_dy) - (-lam*(T[ix,iy,iz] - T[ix,iy-1,iz])*_dy))*_dy
                                                            - ((-lam*(T[ix,iy,iz+1] - T[ix,iy,iz])*_dz) - (-lam*(T[ix,iy,iz] - T[ix,iy,iz-1])*_dz))*_dz)
                                                            );
                                        end
                                        return
                                    end
                                    @parallel memopt=true diffusion3D_step!(T2, T, Ci, lam, dt, _dx, _dy, _dz);
                                    T2_ref[2:end-1,2:end-1,2:end-1] .= T[2:end-1,2:end-1,2:end-1] .+ dt.*(Ci[2:end-1,2:end-1,2:end-1].*(
                                                            - ((.-lam.*(T[3:end  ,2:end-1,2:end-1] .- T[2:end-1,2:end-1,2:end-1]).*_dx) .- (.-lam.*(T[2:end-1,2:end-1,2:end-1] .- T[1:end-2,2:end-1,2:end-1]).*_dx)).*_dx
                                                            - ((.-lam.*(T[2:end-1,3:end  ,2:end-1] .- T[2:end-1,2:end-1,2:end-1]).*_dy) .- (.-lam.*(T[2:end-1,2:end-1,2:end-1] .- T[2:end-1,1:end-2,2:end-1]).*_dy)).*_dy
                                                            - ((.-lam.*(T[2:end-1,2:end-1,3:end  ] .- T[2:end-1,2:end-1,2:end-1]).*_dz) .- (.-lam.*(T[2:end-1,2:end-1,2:end-1] .- T[2:end-1,2:end-1,1:end-2]).*_dz)).*_dz)
                                                            );
                                    @test all(Array(T2) .== Array(T2_ref))
                                end
                                @testset "@parallel <kernel> (3D, memopt, stencilranges=0:2)" begin
                                    lam=dt=_dx=_dy=_dz = 1
                                    T      = @Field(nxyz);
                                    T2     = @Field(nxyz);
                                    T2_ref = @Field(nxyz);
                                    Ci     = @Field(nxyz, @ones);
                                    copy!(T, [ix + (iy-1)*size(T,1) + (iz-1)*size(T,1)*size(T,2) for ix=1:size(T,1), iy=1:size(T,2), iz=1:size(T,3)].^3);
                                    @parallel memopt=true loopsize=3 function diffusion3D_step!(T2, T, Ci, lam, dt, _dx, _dy, _dz)
                                        @inn(T2) = @inn(T) + dt*(lam*@inn(Ci)*(@d2_xi(T)*_dx^2 + @d2_yi(T)*_dy^2 + @d2_zi(T)*_dz^2))
                                        return
                                    end
                                    @parallel memopt=true diffusion3D_step!(T2, T, Ci, lam, dt, _dx, _dy, _dz);
                                    T2_ref[2:end-1,2:end-1,2:end-1] .= T[2:end-1,2:end-1,2:end-1] .+ dt.*(lam.*Ci[2:end-1,2:end-1,2:end-1].*(
                                                            ((T[3:end  ,2:end-1,2:end-1] .- T[2:end-1,2:end-1,2:end-1]) .- (T[2:end-1,2:end-1,2:end-1] .- T[1:end-2,2:end-1,2:end-1])).*_dx^2
                                                            + ((T[2:end-1,3:end  ,2:end-1] .- T[2:end-1,2:end-1,2:end-1]) .- (T[2:end-1,2:end-1,2:end-1] .- T[2:end-1,1:end-2,2:end-1])).*_dy^2
                                                            + ((T[2:end-1,2:end-1,3:end  ] .- T[2:end-1,2:end-1,2:end-1]) .- (T[2:end-1,2:end-1,2:end-1] .- T[2:end-1,2:end-1,1:end-2])).*_dz^2)
                                                            );
                                    @test all(Array(T2) .== Array(T2_ref))
                                end
                                @testset "@parallel_indices <kernel> (3D, memopt, stencilranges=(-4:-1, 2:2, -2:3); x-z-stencil, y-shift)" begin
                                    A      = @Field(nxyz);
                                    A2     = @Field(nxyz);
                                    A2_ref = @Field(nxyz);
                                    copy!(A, [ix + (iy-1)*size(A,1) + (iz-1)*size(A,1)*size(A,2) for ix=1:size(A,1), iy=1:size(A,2), iz=1:size(A,3)].^3);
                                    @parallel_indices (ix,iy,iz) memopt=true loopsize=3 function higher_order_memopt!(A2, A)
                                        if (ix-4>1 && ix-1<size(A2,1) && iy+2>1 && iy+2<=size(A2,2) && iz-2>=1 && iz+3<=size(A2,3))
                                            A2[ix-1,iy+2,iz] = A[ix-1,iy+2,iz+3] - 2*A[ix-3,iy+2,iz] + A[ix-4,iy+2,iz-2]
                                        end
                                        return
                                    end
                                    @parallel memopt=true higher_order_memopt!(A2, A);
                                    A2_ref[5:end-1,3:end,3:end-3] .= A[5:end-1,3:end,6:end] .- 2*A[3:end-3,3:end,3:end-3] .+ A[2:end-4,3:end,1:end-5];
                                    @test all(Array(A2) .== Array(A2_ref))
                                end
                                @testset "@parallel <kernel> (3D, memopt, stencilranges=0:2; on-the-fly)" begin
                                    lam=dt=_dx=_dy=_dz = $FloatDefault(1)
                                    T      = @Field(nxyz);
                                    T2     = @Field(nxyz);
                                    T2_ref = @Field(nxyz);
                                    Ci     = @Field(nxyz, @ones);
                                    copy!(T, [ix + (iy-1)*size(T,1) + (iz-1)*size(T,1)*size(T,2) for ix=1:size(T,1), iy=1:size(T,2), iz=1:size(T,3)].^3);
                                    @parallel memopt=true loopsize=3 function diffusion3D_step!(T2, T, Ci, lam::Data.Number, dt::$FloatDefault, _dx, _dy, _dz)
                                        @all(qx)   = -lam*@d_xi(T)*_dx                                          # Fourier's law of heat conduction
                                        @all(qy)   = -lam*@d_yi(T)*_dy                                          # ...
                                        @all(qz)   = -lam*@d_zi(T)*_dz                                          # ...
                                        @all(dTdt) = @inn(Ci)*(-@d_xa(qx)*_dx - @d_ya(qy)*_dy - @d_za(qz)*_dz)  # Conservation of energy
                                        @inn(T2)   = @inn(T) + dt*@all(dTdt)                                    # Update of temperature
                                        return
                                    end
                                    @parallel memopt=true diffusion3D_step!(T2, T, Ci, lam, dt, _dx, _dy, _dz);
                                    T2_ref[2:end-1,2:end-1,2:end-1] .= T[2:end-1,2:end-1,2:end-1] .+ dt.*(lam.*Ci[2:end-1,2:end-1,2:end-1].*(
                                                            ((T[3:end  ,2:end-1,2:end-1] .- T[2:end-1,2:end-1,2:end-1]) .- (T[2:end-1,2:end-1,2:end-1] .- T[1:end-2,2:end-1,2:end-1])).*_dx^2
                                                            + ((T[2:end-1,3:end  ,2:end-1] .- T[2:end-1,2:end-1,2:end-1]) .- (T[2:end-1,2:end-1,2:end-1] .- T[2:end-1,1:end-2,2:end-1])).*_dy^2
                                                            + ((T[2:end-1,2:end-1,3:end  ] .- T[2:end-1,2:end-1,2:end-1]) .- (T[2:end-1,2:end-1,2:end-1] .- T[2:end-1,2:end-1,1:end-2])).*_dz^2)
                                                            );
                                    @test all(Array(T2) .== Array(T2_ref))
                                end
                                @testset "@parallel <kernel> (3D, memopt, stencilranges=0:0; 2 arrays)" begin
                                    A  = @Field(nxyz);
                                    A2 = @Field(nxyz);
                                    B  = @Field(nxyz);
                                    copy!(A, [ix + (iy-1)*size(A,1) + (iz-1)*size(A,1)*size(A,2) for ix=1:size(A,1), iy=1:size(A,2), iz=1:size(A,3)].^3);
                                    copy!(B, 2 .* [ix + (iy-1)*size(B,1) + (iz-1)*size(B,1)*size(B,2) for ix=1:size(B,1), iy=1:size(B,2), iz=1:size(B,3)].^3);
                                    @parallel memopt=true loopsize=3 optvars=(A, B) optranges=(A=(0:0,0:0,0:0), B=(0:0,0:0,0:0)) function copy_memopt!(A2, A, B)
                                        @all(A2) = @all(A) + @all(B)
                                        return
                                    end
                                    @parallel memopt=true copy_memopt!(A2, A, B);
                                    @test all(Array(A2) .== Array(A) .+ Array(B))
                                end                        
                                @testset "@parallel_indices <kernel> (3D, memopt; 2 arrays, z-stencil)" begin
                                    A      = @Field(nxyz);
                                    A2     = @Field(nxyz);
                                    A2_ref = @Field(nxyz);
                                    B      = @XXYYZField(nxyz);
                                    copy!(A, [ix + (iy-1)*size(A,1) + (iz-1)*size(A,1)*size(A,2) for ix=1:size(A,1), iy=1:size(A,2), iz=1:size(A,3)].^3);
                                    copy!(B, 2 .* [ix + (iy-1)*size(B,1) + (iz-1)*size(B,1)*size(B,2) for ix=1:size(B,1), iy=1:size(B,2), iz=1:size(B,3)].^3);
                                    @parallel_indices (ix,iy,iz) memopt=true loopsize=3 function d2_memopt!(A2, A, B)
                                        if (iz>1 && iz<size(A2,3))
                                            A2[ix,iy,iz] = A[ix,iy,iz+1] - 2*A[ix,iy,iz] + A[ix,iy,iz-1] + B[ix,iy,iz] - B[ix,iy,iz-1]
                                        end
                                        return
                                    end
                                    @parallel memopt=true d2_memopt!(A2, A, B);
                                    A2_ref[:,:,2:end-1] .= A[:,:,3:end] .- 2*A[:,:,2:end-1] .+ A[:,:,1:end-2] .+ B[:,:,2:end] .- B[:,:,1:end-1];
                                    @test all(Array(A2) .== Array(A2_ref))
                                end
                                @testset "@parallel_indices <kernel> (3D, memopt; 2 arrays, y-stencil)" begin
                                    A      = @Field(nxyz);
                                    A2     = @Field(nxyz);
                                    A2_ref = @Field(nxyz);
                                    B      = @Field(nxyz);
                                    copy!(A, [ix + (iy-1)*size(A,1) + (iz-1)*size(A,1)*size(A,2) for ix=1:size(A,1), iy=1:size(A,2), iz=1:size(A,3)].^3);
                                    copy!(B, 2 .* [ix + (iy-1)*size(B,1) + (iz-1)*size(B,1)*size(B,2) for ix=1:size(B,1), iy=1:size(B,2), iz=1:size(B,3)].^3);
                                    @parallel_indices (ix,iy,iz) memopt=true loopsize=3 function d2_memopt!(A2, A, B)
                                        if (iy>1 && iy<size(A2,2))
                                            A2[ix,iy,iz] = A[ix,iy+1,iz] - 2*A[ix,iy,iz] + A[ix,iy-1,iz] + B[ix,iy+1,iz] - 2*B[ix,iy,iz] + B[ix,iy-1,iz]
                                        end
                                        return
                                    end
                                    @parallel memopt=true d2_memopt!(A2, A, B);
                                    A2_ref[:,2:end-1,:] .= (((A[:,3:end,:] .- 2*A[:,2:end-1,:]) .+ A[:,1:end-2,:] .+ B[:,3:end,:]) .- 2*B[:,2:end-1,:]) .+ B[:,1:end-2,:];
                                    @test all(Array(A2) .== Array(A2_ref))
                                end
                                @testset "@parallel_indices <kernel> (3D, memopt; 2 arrays, x-stencil)" begin
                                    A      = @Field(nxyz);
                                    A2     = @Field(nxyz);
                                    A2_ref = @Field(nxyz);
                                    B      = @XYYZZField(nxyz);
                                    copy!(A, [ix + (iy-1)*size(A,1) + (iz-1)*size(A,1)*size(A,2) for ix=1:size(A,1), iy=1:size(A,2), iz=1:size(A,3)].^3);
                                    copy!(B, 2 .* [ix + (iy-1)*size(B,1) + (iz-1)*size(B,1)*size(B,2) for ix=1:size(B,1), iy=1:size(B,2), iz=1:size(B,3)].^3);
                                    @parallel_indices (ix,iy,iz) memopt=true function d2_memopt!(A2, A, B)
                                        if (ix>1 && ix<size(A2,1))
                                            A2[ix,iy,iz] = A[ix+1,iy,iz] - 2*A[ix,iy,iz] + A[ix-1,iy,iz] + B[ix,iy,iz] - B[ix-1,iy,iz]
                                        end
                                        return
                                    end
                                    @parallel memopt=true d2_memopt!(A2, A, B);
                                    A2_ref[2:end-1,:,:] .= A[3:end,:,:] .- 2*A[2:end-1,:,:] .+ A[1:end-2,:,:] .+ B[2:end,:,:] .- B[1:end-1,:,:];
                                    @test all(Array(A2) .== Array(A2_ref))
                                end
                                @testset "@parallel <kernel> (3D, memopt; 2 arrays, x-y-z- + z-stencil)" begin
                                    lam=dt=_dx=_dy=_dz = $FloatDefault(1)
                                    T      = @Field(nxyz);
                                    T2     = @Field(nxyz);
                                    T2_ref = @Field(nxyz);
                                    Ci     = @XXYYZField(nxyz);
                                    copy!(T, [ix + (iy-1)*size(T,1) + (iz-1)*size(T,1)*size(T,2) for ix=1:size(T,1), iy=1:size(T,2), iz=1:size(T,3)].^3);
                                    copy!(Ci, 2 .* [ix + (iy-1)*size(Ci,1) + (iz-1)*size(Ci,1)*size(Ci,2) for ix=1:size(Ci,1), iy=1:size(Ci,2), iz=1:size(Ci,3)].^3);
                                    @parallel memopt=true loopsize=3 function diffusion3D_step_modified!(T2, T, Ci, lam, dt, _dx, _dy, _dz)
                                        @inn(T2) = @inn(T) + dt*(lam*@d_zi(Ci)*(@d2_xi(T)*_dx^2 + @d2_yi(T)*_dy^2 + @d2_zi(T)*_dz^2))
                                        return
                                    end
                                    @parallel memopt=true diffusion3D_step_modified!(T2, T, Ci, lam, dt, _dx, _dy, _dz);
                                    T2_ref[2:end-1,2:end-1,2:end-1] .= T[2:end-1,2:end-1,2:end-1] .+ dt.*(lam.*(Ci[2:end-1,2:end-1,2:end] .- Ci[2:end-1,2:end-1,1:end-1]).*(
                                                            ((T[3:end  ,2:end-1,2:end-1] .- T[2:end-1,2:end-1,2:end-1]) .- (T[2:end-1,2:end-1,2:end-1] .- T[1:end-2,2:end-1,2:end-1])).*_dx^2
                                                            + ((T[2:end-1,3:end  ,2:end-1] .- T[2:end-1,2:end-1,2:end-1]) .- (T[2:end-1,2:end-1,2:end-1] .- T[2:end-1,1:end-2,2:end-1])).*_dy^2
                                                            + ((T[2:end-1,2:end-1,3:end  ] .- T[2:end-1,2:end-1,2:end-1]) .- (T[2:end-1,2:end-1,2:end-1] .- T[2:end-1,2:end-1,1:end-2])).*_dz^2)
                                                            );
                                    @test all(Array(T2) .== Array(T2_ref))
                                end
                                @testset "@parallel <kernel> (3D, memopt; 2 arrays, x-y-z- + x-stencil)" begin
                                    lam=dt=_dx=_dy=_dz = $FloatDefault(1)
                                    T      = @Field(nxyz);
                                    T2     = @Field(nxyz);
                                    T2_ref = @Field(nxyz);
                                    Ci     = @XYYZZField(nxyz);
                                    copy!(T, [ix + (iy-1)*size(T,1) + (iz-1)*size(T,1)*size(T,2) for ix=1:size(T,1), iy=1:size(T,2), iz=1:size(T,3)].^3);
                                    copy!(Ci, 2 .* [ix + (iy-1)*size(Ci,1) + (iz-1)*size(Ci,1)*size(Ci,2) for ix=1:size(Ci,1), iy=1:size(Ci,2), iz=1:size(Ci,3)].^3);
                                    @parallel memopt=true loopsize=3 function diffusion3D_step_modified!(T2, T, Ci, lam, dt, _dx, _dy, _dz)
                                        @inn(T2) = @inn(T) + dt*(lam*@d_xi(Ci)*(@d2_xi(T)*_dx^2 + @d2_yi(T)*_dy^2 + @d2_zi(T)*_dz^2))
                                        return
                                    end
                                    @parallel memopt=true diffusion3D_step_modified!(T2, T, Ci, lam, dt, _dx, _dy, _dz);
                                    T2_ref[2:end-1,2:end-1,2:end-1] .= T[2:end-1,2:end-1,2:end-1] .+ dt.*(lam.*(Ci[2:end,2:end-1,2:end-1] .- Ci[1:end-1,2:end-1,2:end-1]).*(
                                                            ((T[3:end  ,2:end-1,2:end-1] .- T[2:end-1,2:end-1,2:end-1]) .- (T[2:end-1,2:end-1,2:end-1] .- T[1:end-2,2:end-1,2:end-1])).*_dx^2
                                                            + ((T[2:end-1,3:end  ,2:end-1] .- T[2:end-1,2:end-1,2:end-1]) .- (T[2:end-1,2:end-1,2:end-1] .- T[2:end-1,1:end-2,2:end-1])).*_dy^2
                                                            + ((T[2:end-1,2:end-1,3:end  ] .- T[2:end-1,2:end-1,2:end-1]) .- (T[2:end-1,2:end-1,2:end-1] .- T[2:end-1,2:end-1,1:end-2])).*_dz^2)
                                                            );
                                    @test all(Array(T2) .== Array(T2_ref))
                                end
                                @testset "@parallel <kernel> (3D, memopt; 3 arrays, x-y-z- + y- + x-stencil)" begin
                                    lam=dt=_dx=_dy=_dz = $FloatDefault(1)
                                    T      = @Field(nxyz);
                                    T2     = @Field(nxyz);
                                    T2_ref = @Field(nxyz);
                                    Ci     = @XYYZZField(nxyz);
                                    B      = @Field(nxyz);
                                    copy!(T, [ix + (iy-1)*size(T,1) + (iz-1)*size(T,1)*size(T,2) for ix=1:size(T,1), iy=1:size(T,2), iz=1:size(T,3)].^3);
                                    copy!(Ci, 2 .* [ix + (iy-1)*size(Ci,1) + (iz-1)*size(Ci,1)*size(Ci,2) for ix=1:size(Ci,1), iy=1:size(Ci,2), iz=1:size(Ci,3)].^3);
                                    copy!(B,  3 .* [ix + (iy-1)*size(B,1) + (iz-1)*size(B,1)*size(B,2) for ix=1:size(B,1), iy=1:size(B,2), iz=1:size(B,3)].^3);
                                    @parallel memopt=true loopsize=3 function diffusion3D_step_modified!(T2, T, Ci, B, lam, dt, _dx, _dy, _dz)
                                        @inn(T2) = @inn(T) + dt*(lam*@d_xi(Ci)*(@d2_xi(T)*_dx^2 + @d2_yi(T)*_dy^2 + @d2_zi(T)*_dz^2)) + @d2_yi(B)
                                        return
                                    end
                                    @parallel memopt=true diffusion3D_step_modified!(T2, T, Ci, B, lam, dt, _dx, _dy, _dz);
                                    T2_ref[2:end-1,2:end-1,2:end-1] .= T[2:end-1,2:end-1,2:end-1] .+ dt.*(lam.*(Ci[2:end,2:end-1,2:end-1] .- Ci[1:end-1,2:end-1,2:end-1]).*(
                                                            ((T[3:end  ,2:end-1,2:end-1] .- T[2:end-1,2:end-1,2:end-1]) .- (T[2:end-1,2:end-1,2:end-1] .- T[1:end-2,2:end-1,2:end-1])).*_dx^2
                                                            + ((T[2:end-1,3:end  ,2:end-1] .- T[2:end-1,2:end-1,2:end-1]) .- (T[2:end-1,2:end-1,2:end-1] .- T[2:end-1,1:end-2,2:end-1])).*_dy^2
                                                            + ((T[2:end-1,2:end-1,3:end  ] .- T[2:end-1,2:end-1,2:end-1]) .- (T[2:end-1,2:end-1,2:end-1] .- T[2:end-1,2:end-1,1:end-2])).*_dz^2)
                                                            ) + ((B[2:end-1,3:end  ,2:end-1] .- B[2:end-1,2:end-1,2:end-1]) .- (B[2:end-1,2:end-1,2:end-1] .- B[2:end-1,1:end-2,2:end-1]));
                                    @test all(Array(T2) .== Array(T2_ref))
                                end
                                @testset "@parallel_indices <kernel> (3D, memopt, stencilranges=(-4:-1, 2:2, -2:3); 3 arrays, x-z-stencil, y-shift)" begin
                                    A      = @Field(nxyz);
                                    A2     = @Field(nxyz);
                                    A2_ref = @Field(nxyz);
                                    B      = @Field(nxyz);
                                    B2     = @Field(nxyz);
                                    B2_ref = @Field(nxyz);
                                    C      = @Field(nxyz);
                                    C2     = @Field(nxyz);
                                    C2_ref = @Field(nxyz);
                                    copy!(A, [ix + (iy-1)*size(A,1) + (iz-1)*size(A,1)*size(A,2) for ix=1:size(A,1), iy=1:size(A,2), iz=1:size(A,3)].^3);
                                    copy!(B, 2 .* [ix + (iy-1)*size(A,1) + (iz-1)*size(A,1)*size(A,2) for ix=1:size(A,1), iy=1:size(A,2), iz=1:size(A,3)].^3);
                                    copy!(C, 3 .* [ix + (iy-1)*size(A,1) + (iz-1)*size(A,1)*size(A,2) for ix=1:size(A,1), iy=1:size(A,2), iz=1:size(A,3)].^3);
                                    @parallel_indices (ix,iy,iz) memopt=true loopsize=3 function higher_order_memopt!(A2, B2, C2, A, B, C)
                                        if (ix-4>1 && ix-1<size(A2,1) && iy+2>1 && iy+2<=size(A2,2) && iz-2>=1 && iz+3<=size(A2,3))
                                            A2[ix-1,iy+2,iz] = A[ix-1,iy+2,iz+3] - 2*A[ix-3,iy+2,iz] + A[ix-4,iy+2,iz-2]
                                        end
                                        if (ix-4>1 && ix-1<size(B2,1) && iy+2>1 && iy+2<=size(B2,2) && iz-2>=1 && iz+3<=size(B2,3))
                                            B2[ix-1,iy+2,iz] = B[ix-1,iy+2,iz+3] - 2*B[ix-3,iy+2,iz] + B[ix-4,iy+2,iz-2]
                                        end
                                        if (ix-4>1 && ix-1<size(C2,1) && iy+2>1 && iy+2<=size(C2,2) && iz-2>=1 && iz+3<=size(C2,3))
                                            C2[ix-1,iy+2,iz] = C[ix-1,iy+2,iz+3] - 2*C[ix-3,iy+2,iz] + C[ix-4,iy+2,iz-2]
                                        end
                                        return
                                    end
                                    @parallel memopt=true higher_order_memopt!(A2, B2, C2, A, B, C);
                                    A2_ref[5:end-1,3:end,3:end-3] .= A[5:end-1,3:end,6:end] .- 2*A[3:end-3,3:end,3:end-3] .+ A[2:end-4,3:end,1:end-5];
                                    B2_ref[5:end-1,3:end,3:end-3] .= B[5:end-1,3:end,6:end] .- 2*B[3:end-3,3:end,3:end-3] .+ B[2:end-4,3:end,1:end-5];
                                    C2_ref[5:end-1,3:end,3:end-3] .= C[5:end-1,3:end,6:end] .- 2*C[3:end-3,3:end,3:end-3] .+ C[2:end-4,3:end,1:end-5];
                                    @test all(Array(A2) .== Array(A2_ref))
                                    @test all(Array(B2) .== Array(B2_ref))
                                    @test all(Array(C2) .== Array(C2_ref))
                                end
                                @testset "@parallel_indices <kernel> (3D, memopt, stencilranges=(A=(-4:-1, 2:2, -2:3), B=(-4:-1, 2:2, 1:2), C=(-4:-1, 2:2, -1:0)); 3 arrays, x-z-stencil, y-shift)" begin
                                    A      = @Field(nxyz);
                                    A2     = @Field(nxyz);
                                    A2_ref = @Field(nxyz);
                                    B      = @Field(nxyz);
                                    B2     = @Field(nxyz);
                                    B2_ref = @Field(nxyz);
                                    C      = @Field(nxyz);
                                    C2     = @Field(nxyz);
                                    C2_ref = @Field(nxyz);
                                    copy!(A, [ix + (iy-1)*size(A,1) + (iz-1)*size(A,1)*size(A,2) for ix=1:size(A,1), iy=1:size(A,2), iz=1:size(A,3)].^3);
                                    copy!(B, 2 .* [ix + (iy-1)*size(A,1) + (iz-1)*size(A,1)*size(A,2) for ix=1:size(A,1), iy=1:size(A,2), iz=1:size(A,3)].^3);
                                    copy!(C, 3 .* [ix + (iy-1)*size(A,1) + (iz-1)*size(A,1)*size(A,2) for ix=1:size(A,1), iy=1:size(A,2), iz=1:size(A,3)].^3);
                                    @parallel_indices (ix,iy,iz) memopt=true loopsize=3 function higher_order_memopt!(A2, B2, C2, A, B, C)
                                        if (ix-4>1 && ix-1<size(A2,1) && iy+2>1 && iy+2<=size(A2,2) && iz-2>=1 && iz+3<=size(A2,3))
                                            A2[ix-1,iy+2,iz] = A[ix-1,iy+2,iz+3] - 2*A[ix-3,iy+2,iz] + A[ix-4,iy+2,iz-2]
                                        end
                                        if (ix-4>1 && ix-1<size(B2,1) && iy+2>1 && iy+2<=size(B2,2) && iz+1>=1 && iz+2<=size(B2,3))
                                            B2[ix-1,iy+2,iz+1] = B[ix-1,iy+2,iz+2] - 2*B[ix-3,iy+2,iz+1] + B[ix-4,iy+2,iz+1]
                                        end
                                        if (ix-4>1 && ix-1<size(C2,1) && iy+2>1 && iy+2<=size(C2,2) && iz-1>=1 && iz<=size(C2,3))
                                            C2[ix-1,iy+2,iz-1] = C[ix-1,iy+2,iz] - 2*C[ix-3,iy+2,iz-1] + C[ix-4,iy+2,iz-1]
                                        end
                                        return
                                    end
                                    @parallel memopt=true higher_order_memopt!(A2, B2, C2, A, B, C);
                                    A2_ref[5:end-1,3:end,3:end-3] .= A[5:end-1,3:end,6:end] .- 2*A[3:end-3,3:end,3:end-3] .+ A[2:end-4,3:end,1:end-5];
                                    B2_ref[5:end-1,3:end,2:end-1] .= B[5:end-1,3:end,3:end] .- 2*B[3:end-3,3:end,2:end-1] .+ B[2:end-4,3:end,2:end-1];
                                    C2_ref[5:end-1,3:end,1:end-1] .= C[5:end-1,3:end,2:end] .- 2*C[3:end-3,3:end,1:end-1] .+ C[2:end-4,3:end,1:end-1];
                                    @test all(Array(A2) .== Array(A2_ref))
                                    @test all(Array(B2) .== Array(B2_ref))
                                    @test all(Array(C2) .== Array(C2_ref))
                                end
                                @testset "@parallel_indices <kernel> (3D, memopt, stencilranges=(A=(-4:-1, 2:2, -2:3), B=(-4:1, 2:2, 1:2), C=(-1:-1, 2:2, -1:0)); 3 arrays, x-z-stencil, y-shift)" begin
                                    A      = @Field(nxyz);
                                    A2     = @Field(nxyz);
                                    A2_ref = @Field(nxyz);
                                    B      = @Field(nxyz);
                                    B2     = @Field(nxyz);
                                    B2_ref = @Field(nxyz);
                                    C      = @Field(nxyz);
                                    C2     = @Field(nxyz);
                                    C2_ref = @Field(nxyz);
                                    copy!(A, [ix + (iy-1)*size(A,1) + (iz-1)*size(A,1)*size(A,2) for ix=1:size(A,1), iy=1:size(A,2), iz=1:size(A,3)].^3);
                                    copy!(B, 2 .* [ix + (iy-1)*size(A,1) + (iz-1)*size(A,1)*size(A,2) for ix=1:size(A,1), iy=1:size(A,2), iz=1:size(A,3)].^3);
                                    copy!(C, 3 .* [ix + (iy-1)*size(A,1) + (iz-1)*size(A,1)*size(A,2) for ix=1:size(A,1), iy=1:size(A,2), iz=1:size(A,3)].^3);
                                    @parallel_indices (ix,iy,iz) memopt=true loopsize=3 function higher_order_memopt!(A2, B2, C2, A, B, C)
                                        if (ix-4>1 && ix-1<size(A2,1) && iy+2>1 && iy+2<=size(A2,2) && iz-2>=1 && iz+3<=size(A2,3))
                                            A2[ix-1,iy+2,iz] = A[ix-1,iy+2,iz+3] - 2*A[ix-3,iy+2,iz] + A[ix-4,iy+2,iz-2]
                                        end
                                        if (ix-4>1 && ix+1<size(B2,1) && iy+2>1 && iy+2<=size(B2,2) && iz+1>=1 && iz+2<=size(B2,3))
                                            B2[ix+1,iy+2,iz+1] = B[ix+1,iy+2,iz+2] - 2*B[ix-3,iy+2,iz+1] + B[ix-4,iy+2,iz+1]
                                        end
                                        if (ix-1>1 && ix-1<size(C2,1) && iy+2>1 && iy+2<=size(C2,2) && iz-1>=1 && iz<=size(C2,3))
                                            C2[ix-1,iy+2,iz-1] = C[ix-1,iy+2,iz] - 2*C[ix-1,iy+2,iz-1] + C[ix-1,iy+2,iz-1]
                                        end
                                        return
                                    end
                                    @parallel memopt=true higher_order_memopt!(A2, B2, C2, A, B, C);
                                    A2_ref[5:end-1,3:end,3:end-3] .= A[5:end-1,3:end,6:end] .- 2*A[3:end-3,3:end,3:end-3] .+ A[2:end-4,3:end,1:end-5];
                                    B2_ref[7:end-1,3:end,2:end-1] .= B[7:end-1,3:end,3:end] .- 2*B[3:end-5,3:end,2:end-1] .+ B[2:end-6,3:end,2:end-1];
                                    C2_ref[2:end-1,3:end,1:end-1] .= C[2:end-1,3:end,2:end] .- 2*C[2:end-1,3:end,1:end-1] .+ C[2:end-1,3:end,1:end-1];
                                    @test all(Array(A2) .== Array(A2_ref))
                                    @test all(Array(B2) .== Array(B2_ref))
                                    @test all(Array(C2) .== Array(C2_ref))
                                end
                                @testset "@parallel_indices <kernel> (3D, memopt, optvars=(A, C), loopdim=3, loopsize=3, optranges=(A=(-4:-1, 2:2, -2:3), B=(-4:1, 2:2, 1:2), C=(-1:-1, 2:2, -1:0)); stencilranges=(A=(-4:-1, 2:2, -2:3), B=(-4:1, 2:2, 1:2), C=(-1:-1, 2:2, -1:0)), 3 arrays, x-z-stencil, y-shift)" begin
                                    A      = @Field(nxyz);
                                    A2     = @Field(nxyz);
                                    A2_ref = @Field(nxyz);
                                    B      = @Field(nxyz);
                                    B2     = @Field(nxyz);
                                    B2_ref = @Field(nxyz);
                                    C      = @Field(nxyz);
                                    C2     = @Field(nxyz);
                                    C2_ref = @Field(nxyz);
                                    copy!(A, [ix + (iy-1)*size(A,1) + (iz-1)*size(A,1)*size(A,2) for ix=1:size(A,1), iy=1:size(A,2), iz=1:size(A,3)].^3);
                                    copy!(B, 2 .* [ix + (iy-1)*size(A,1) + (iz-1)*size(A,1)*size(A,2) for ix=1:size(A,1), iy=1:size(A,2), iz=1:size(A,3)].^3);
                                    copy!(C, 3 .* [ix + (iy-1)*size(A,1) + (iz-1)*size(A,1)*size(A,2) for ix=1:size(A,1), iy=1:size(A,2), iz=1:size(A,3)].^3);
                                    kernel = @gorgeousstring @parallel_indices (ix,iy,iz) memopt=true optvars=(A, C) loopdim=3 loopsize=3 optranges=(A=(-4:-1, 2:2, -2:3), B=(-4:1, 2:2, 1:2), C=(-1:-1, 2:2, -1:0)) function higher_order_memopt!(A2, B2, C2, A, B, C)
                                        if (ix-4>1 && ix-1<size(A2,1) && iy+2>1 && iy+2<=size(A2,2) && iz-2>=1 && iz+3<=size(A2,3))
                                            A2[ix-1,iy+2,iz] = A[ix-1,iy+2,iz+3] - 2*A[ix-3,iy+2,iz] + A[ix-4,iy+2,iz-2]
                                        end
                                        if (ix-4>1 && ix+1<size(B2,1) && iy+2>1 && iy+2<=size(B2,2) && iz+1>=1 && iz+2<=size(B2,3))
                                            B2[ix+1,iy+2,iz+1] = B[ix+1,iy+2,iz+2] - 2*B[ix-3,iy+2,iz+1] + B[ix-4,iy+2,iz+1]
                                        end
                                        if (ix-1>1 && ix-1<size(C2,1) && iy+2>1 && iy+2<=size(C2,2) && iz-1>=1 && iz<=size(C2,3))
                                            C2[ix-1,iy+2,iz-1] = C[ix-1,iy+2,iz] - 2*C[ix-1,iy+2,iz-1] + C[ix-1,iy+2,iz-1]
                                        end
                                        return
                                    end
                                    @static if $package == $PKG_CUDA
                                        @test occursin("loopoffset = (((CUDA.blockIdx()).z - 1) * 3 + (ranges[3])[1]) - 1", kernel) # Alternative: @test occursin("loopoffset = ((CUDA.blockIdx()).z - 1) * 3", kernel)
                                    elseif $package == $PKG_AMDGPU
                                        @test occursin("loopoffset = (((AMDGPU.workgroupIdx()).z - 1) * 3 + (ranges[3])[1]) - 1", kernel) # Alternative: @test occursin("loopoffset = ((AMDGPU.workgroupIdx()).z - 1) * 3", kernel)
                                    elseif $package == $PKG_METAL
                                        @test occursin("loopoffset = (((Metal.threadgroup_position_in_grid_3d()).z - 1) * 3 + (ranges[3])[1]) - 1", kernel) # Alternative: @test occursin("loopoffset = ((Metal.threadgroup_position_in_grid_3d()).z - 1) * 3", kernel)
                                    end
                                    @test occursin("for i = -4:3", kernel)
                                    @test occursin("iz = i + loopoffset", kernel) # Alternative: @test occursin("tz = i + loopoffset", kernel)
                                    @test occursin("A2[ix - 1, iy + 2, iz] = (A_ixm1_iyp2_izp3 - 2A_ixm3_iyp2_iz) + A_ixm4_iyp2_izm2", kernel)
                                    @test occursin("B2[ix + 1, iy + 2, iz + 1] = (B[ix + 1, iy + 2, iz + 2] - 2 * B[ix - 3, iy + 2, iz + 1]) + B[ix - 4, iy + 2, iz + 1]", kernel)
                                    @test occursin("C2[ix - 1, iy + 2, iz - 1] = (C_ixm1_iyp2_iz - 2C_ixm1_iyp2_izm1) + C_ixm1_iyp2_izm1", kernel)
                                    @parallel_indices (ix,iy,iz) memopt=true optvars=(A, C) loopdim=3 loopsize=3 optranges=(A=(-4:-1, 2:2, -2:3), B=(-4:1, 2:2, 1:2), C=(-1:-1, 2:2, -1:0)) function higher_order_memopt!(A2, B2, C2, A, B, C)
                                        if (ix-4>1 && ix-1<size(A2,1) && iy+2>1 && iy+2<=size(A2,2) && iz-2>=1 && iz+3<=size(A2,3))
                                            A2[ix-1,iy+2,iz] = A[ix-1,iy+2,iz+3] - 2*A[ix-3,iy+2,iz] + A[ix-4,iy+2,iz-2]
                                        end
                                        if (ix-4>1 && ix+1<size(B2,1) && iy+2>1 && iy+2<=size(B2,2) && iz+1>=1 && iz+2<=size(B2,3))
                                            B2[ix+1,iy+2,iz+1] = B[ix+1,iy+2,iz+2] - 2*B[ix-3,iy+2,iz+1] + B[ix-4,iy+2,iz+1]
                                        end
                                        if (ix-1>1 && ix-1<size(C2,1) && iy+2>1 && iy+2<=size(C2,2) && iz-1>=1 && iz<=size(C2,3))
                                            C2[ix-1,iy+2,iz-1] = C[ix-1,iy+2,iz] - 2*C[ix-1,iy+2,iz-1] + C[ix-1,iy+2,iz-1]
                                        end
                                        return
                                    end
                                    @parallel memopt=true higher_order_memopt!(A2, B2, C2, A, B, C);
                                    A2_ref[5:end-1,3:end,3:end-3] .= A[5:end-1,3:end,6:end] .- 2*A[3:end-3,3:end,3:end-3] .+ A[2:end-4,3:end,1:end-5];
                                    B2_ref[7:end-1,3:end,2:end-1] .= B[7:end-1,3:end,3:end] .- 2*B[3:end-5,3:end,2:end-1] .+ B[2:end-6,3:end,2:end-1];
                                    C2_ref[2:end-1,3:end,1:end-1] .= C[2:end-1,3:end,2:end] .- 2*C[2:end-1,3:end,1:end-1] .+ C[2:end-1,3:end,1:end-1];
                                    @test all(Array(A2) .== Array(A2_ref))
                                    @test all(Array(B2) .== Array(B2_ref))
                                    @test all(Array(C2) .== Array(C2_ref))
                                end
                                @testset "@parallel_indices <kernel> (3D, memopt, optvars=(A, C), loopdim=3, loopsize=3, optranges=(A=(-4:-1, 2:2, -2:3), B=(-4:1, 2:2, 1:2), C=(-1:-1, 2:2, -1:0)); stencilranges=(A=(-4:-1, 2:2, -2:3), B=(-4:1, 2:2, 1:2), C=(-1:-1, 2:2, -1:0)), 3 arrays, x-z-stencil, y-shift)" begin
                                    A      = @Field(nxyz);
                                    A2     = @Field(nxyz);
                                    A2_ref = @Field(nxyz);
                                    B      = @Field(nxyz);
                                    B2     = @Field(nxyz);
                                    B2_ref = @Field(nxyz);
                                    C      = @Field(nxyz);
                                    C2     = @Field(nxyz);
                                    C2_ref = @Field(nxyz);
                                    copy!(A, [ix + (iy-1)*size(A,1) + (iz-1)*size(A,1)*size(A,2) for ix=1:size(A,1), iy=1:size(A,2), iz=1:size(A,3)].^3);
                                    copy!(B, 2 .* [ix + (iy-1)*size(A,1) + (iz-1)*size(A,1)*size(A,2) for ix=1:size(A,1), iy=1:size(A,2), iz=1:size(A,3)].^3);
                                    copy!(C, 3 .* [ix + (iy-1)*size(A,1) + (iz-1)*size(A,1)*size(A,2) for ix=1:size(A,1), iy=1:size(A,2), iz=1:size(A,3)].^3);
                                    kernel = @gorgeousstring @parallel_indices (ix,iy,iz) memopt=true optvars=(A, C) loopdim=3 loopsize=3 optranges=(A=(-4:-1, 2:2, -2:3), B=(-4:1, 2:2, 1:2), C=(-1:-1, 2:2, -1:0)) function higher_order_memopt!(A2, B2, C2, A, B, C)
                                        if (ix-4>1 && ix-1<size(A2,1) && iy+2>1 && iy+2<=size(A2,2) && iz-2>=1 && iz+3<=size(A2,3))
                                            A2[ix-1,iy+2,iz] = A[ix-1,iy+2,iz+3] - 2*A[ix-3,iy+2,iz] + A[ix-4,iy+2,iz-2]
                                        end
                                        if (ix-4>1 && ix+1<size(B2,1) && iy+2>1 && iy+2<=size(B2,2) && iz+1>=1 && iz+2<=size(B2,3))
                                            B2[ix+1,iy+2,iz+1] = B[ix+1,iy+2,iz+2] - 2*B[ix-3,iy+2,iz+1] + B[ix-4,iy+2,iz+1]
                                        end
                                        if (ix-1>1 && ix-1<size(C2,1) && iy+2>1 && iy+2<=size(C2,2) && iz-1>=1 && iz<=size(C2,3))
                                            C2[ix-1,iy+2,iz-1] = C[ix-1,iy+2,iz] - 2*C[ix-1,iy+2,iz-1] + C[ix-1,iy+2,iz-1]
                                        end
                                        return
                                    end
                                    @static if $package == $PKG_CUDA
                                        @test occursin("loopoffset = (((CUDA.blockIdx()).z - 1) * 3 + (ranges[3])[1]) - 1", kernel) # Alternative: @test occursin("loopoffset = ((CUDA.blockIdx()).z - 1) * 3", kernel)
                                    elseif $package == $PKG_AMDGPU
                                        @test occursin("loopoffset = (((AMDGPU.workgroupIdx()).z - 1) * 3 + (ranges[3])[1]) - 1", kernel) # Alternative: @test occursin("loopoffset = ((AMDGPU.workgroupIdx()).z - 1) * 3", kernel)
                                    elseif $package == $PKG_METAL
                                        @test occursin("loopoffset = (((Metal.threadgroup_position_in_grid_3d()).z - 1) * 3 + (ranges[3])[1]) - 1", kernel) # Alternative: @test occursin("loopoffset = ((Metal.threadgroup_position_in_grid_3d()).z - 1) * 3", kernel)
                                    end
                                    @test occursin("for i = -4:3", kernel)
                                    @test occursin("iz = i + loopoffset", kernel) # Alternative: @test occursin("tz = i + loopoffset", kernel)
                                    @test occursin("A2[ix - 1, iy + 2, iz] = (A_ixm1_iyp2_izp3 - 2A_ixm3_iyp2_iz) + A_ixm4_iyp2_izm2", kernel)
                                    @test occursin("B2[ix + 1, iy + 2, iz + 1] = (B[ix + 1, iy + 2, iz + 2] - 2 * B[ix - 3, iy + 2, iz + 1]) + B[ix - 4, iy + 2, iz + 1]", kernel)
                                    @test occursin("C2[ix - 1, iy + 2, iz - 1] = (C_ixm1_iyp2_iz - 2C_ixm1_iyp2_izm1) + C_ixm1_iyp2_izm1", kernel)
                                    @parallel_indices (ix,iy,iz) memopt=true optvars=(A, C) loopdim=3 loopsize=3 optranges=(A=(-4:-1, 2:2, -2:3), B=(-4:1, 2:2, 1:2), C=(-1:-1, 2:2, -1:0)) function higher_order_memopt!(A2, B2, C2, A, B, C)
                                        if (ix-4>1 && ix-1<size(A2,1) && iy+2>1 && iy+2<=size(A2,2) && iz-2>=1 && iz+3<=size(A2,3))
                                            A2[ix-1,iy+2,iz] = A[ix-1,iy+2,iz+3] - 2*A[ix-3,iy+2,iz] + A[ix-4,iy+2,iz-2]
                                        end
                                        if (ix-4>1 && ix+1<size(B2,1) && iy+2>1 && iy+2<=size(B2,2) && iz+1>=1 && iz+2<=size(B2,3))
                                            B2[ix+1,iy+2,iz+1] = B[ix+1,iy+2,iz+2] - 2*B[ix-3,iy+2,iz+1] + B[ix-4,iy+2,iz+1]
                                        end
                                        if (ix-1>1 && ix-1<size(C2,1) && iy+2>1 && iy+2<=size(C2,2) && iz-1>=1 && iz<=size(C2,3))
                                            C2[ix-1,iy+2,iz-1] = C[ix-1,iy+2,iz] - 2*C[ix-1,iy+2,iz-1] + C[ix-1,iy+2,iz-1]
                                        end
                                        return
                                    end
                                    @parallel memopt=true higher_order_memopt!(A2, B2, C2, A, B, C);
                                    A2_ref[5:end-1,3:end,3:end-3] .= A[5:end-1,3:end,6:end] .- 2*A[3:end-3,3:end,3:end-3] .+ A[2:end-4,3:end,1:end-5];
                                    B2_ref[7:end-1,3:end,2:end-1] .= B[7:end-1,3:end,3:end] .- 2*B[3:end-5,3:end,2:end-1] .+ B[2:end-6,3:end,2:end-1];
                                    C2_ref[2:end-1,3:end,1:end-1] .= C[2:end-1,3:end,2:end] .- 2*C[2:end-1,3:end,1:end-1] .+ C[2:end-1,3:end,1:end-1];
                                    @test all(Array(A2) .== Array(A2_ref))
                                    @test all(Array(B2) .== Array(B2_ref))
                                    @test all(Array(C2) .== Array(C2_ref))
                                end
                                @testset "@parallel_indices <kernel> (3D, memopt, optvars=(A, B), loopdim=3, loopsize=3, optranges=(A=(-1:-1, 2:2, -2:3), B=(-4:1, 2:2, 1:1)); stencilranges=(A=(-4:-1, 2:2, -2:3), B=(-4:1, 2:2, 1:2), C=(-1:-1, 2:2, -1:0)), 3 arrays, x-z-stencil, y-shift)" begin
                                    A      = @Field(nxyz);
                                    A2     = @Field(nxyz);
                                    A2_ref = @Field(nxyz);
                                    B      = @Field(nxyz);
                                    B2     = @Field(nxyz);
                                    B2_ref = @Field(nxyz);
                                    C      = @Field(nxyz);
                                    C2     = @Field(nxyz);
                                    C2_ref = @Field(nxyz);
                                    copy!(A, [ix + (iy-1)*size(A,1) + (iz-1)*size(A,1)*size(A,2) for ix=1:size(A,1), iy=1:size(A,2), iz=1:size(A,3)].^3);
                                    copy!(B, 2 .* [ix + (iy-1)*size(A,1) + (iz-1)*size(A,1)*size(A,2) for ix=1:size(A,1), iy=1:size(A,2), iz=1:size(A,3)].^3);
                                    copy!(C, 3 .* [ix + (iy-1)*size(A,1) + (iz-1)*size(A,1)*size(A,2) for ix=1:size(A,1), iy=1:size(A,2), iz=1:size(A,3)].^3);
                                    kernel = @gorgeousstring @parallel_indices (ix,iy,iz) memopt=true optvars=(A, B) loopdim=3 loopsize=3 optranges=(A=(-1:-1, 2:2, -2:3), B=(-4:-3, 2:2, 1:1)) function higher_order_memopt!(A2, B2, C2, A, B, C)
                                        if (ix-4>1 && ix-1<size(A2,1) && iy+2>1 && iy+2<=size(A2,2) && iz-2>=1 && iz+3<=size(A2,3))
                                            A2[ix-1,iy+2,iz] = A[ix-1,iy+2,iz+3] - 2*A[ix-3,iy+2,iz] + A[ix-4,iy+2,iz-2]
                                        end
                                        if (ix-4>1 && ix+1<size(B2,1) && iy+2>1 && iy+2<=size(B2,2) && iz+1>=1 && iz+2<=size(B2,3))
                                            B2[ix+1,iy+2,iz+1] = B[ix+1,iy+2,iz+2] - 2*B[ix-3,iy+2,iz+1] + B[ix-4,iy+2,iz+1]
                                        end
                                        if (ix-1>1 && ix-1<size(C2,1) && iy+2>1 && iy+2<=size(C2,2) && iz-1>=1 && iz<=size(C2,3))
                                            C2[ix-1,iy+2,iz-1] = C[ix-1,iy+2,iz] - 2*C[ix-1,iy+2,iz-1] + C[ix-1,iy+2,iz-1]
                                        end
                                        return
                                    end
                                    @test occursin("A2[ix - 1, iy + 2, iz] = (A_ixm1_iyp2_izp3 - 2 * A[ix - 3, iy + 2, iz]) + A[ix - 4, iy + 2, iz - 2]", kernel)
                                    @test occursin("B2[ix + 1, iy + 2, iz + 1] = (B[ix + 1, iy + 2, iz + 2] - 2B_ixm3_iyp2_izp1) + B_ixm4_iyp2_izp1", kernel) # NOTE: when z is restricted to 1:1 then x cannot include +1, as else the x-y range does not include any z (result: IncoherentArgumentError: incoherent argument in memopt: optranges in z dimension do not include any array access.).
                                    @test occursin("C2[ix - 1, iy + 2, iz - 1] = (C[ix - 1, iy + 2, iz] - 2 * C[ix - 1, iy + 2, iz - 1]) + C[ix - 1, iy + 2, iz - 1]", kernel)
                                    @parallel_indices (ix,iy,iz) memopt=true optvars=(A, B) loopdim=3 loopsize=3 optranges=(A=(-1:-1, 2:2, -2:3), B=(-4:-3, 2:2, 1:1)) function higher_order_memopt!(A2, B2, C2, A, B, C)
                                        if (ix-4>1 && ix-1<size(A2,1) && iy+2>1 && iy+2<=size(A2,2) && iz-2>=1 && iz+3<=size(A2,3))
                                            A2[ix-1,iy+2,iz] = A[ix-1,iy+2,iz+3] - 2*A[ix-3,iy+2,iz] + A[ix-4,iy+2,iz-2]
                                        end
                                        if (ix-4>1 && ix+1<size(B2,1) && iy+2>1 && iy+2<=size(B2,2) && iz+1>=1 && iz+2<=size(B2,3))
                                            B2[ix+1,iy+2,iz+1] = B[ix+1,iy+2,iz+2] - 2*B[ix-3,iy+2,iz+1] + B[ix-4,iy+2,iz+1]
                                        end
                                        if (ix-1>1 && ix-1<size(C2,1) && iy+2>1 && iy+2<=size(C2,2) && iz-1>=1 && iz<=size(C2,3))
                                            C2[ix-1,iy+2,iz-1] = C[ix-1,iy+2,iz] - 2*C[ix-1,iy+2,iz-1] + C[ix-1,iy+2,iz-1]
                                        end
                                        return
                                    end
                                    @parallel memopt=true higher_order_memopt!(A2, B2, C2, A, B, C);
                                    A2_ref[5:end-1,3:end,3:end-3] .= A[5:end-1,3:end,6:end] .- 2*A[3:end-3,3:end,3:end-3] .+ A[2:end-4,3:end,1:end-5];
                                    B2_ref[7:end-1,3:end,2:end-1] .= B[7:end-1,3:end,3:end] .- 2*B[3:end-5,3:end,2:end-1] .+ B[2:end-6,3:end,2:end-1];
                                    C2_ref[2:end-1,3:end,1:end-1] .= C[2:end-1,3:end,2:end] .- 2*C[2:end-1,3:end,1:end-1] .+ C[2:end-1,3:end,1:end-1];
                                    @test all(Array(A2) .== Array(A2_ref))
                                    @test all(Array(B2) .== Array(B2_ref))
                                    @test all(Array(C2) .== Array(C2_ref))
                                end
                            end
                            @testset "@parallel memopt <kernel> (nx, ny, nz != x .* threads)" begin
                                nxyz = (33, 7, 8)
                                @testset "@parallel_indices <kernel> (3D, memopt, stencilranges=0:0)" begin
                                    A  = @Field(nxyz);
                                    A2 = @Field(nxyz);
                                    copy!(A, [ix + (iy-1)*size(A,1) + (iz-1)*size(A,1)*size(A,2) for ix=1:size(A,1), iy=1:size(A,2), iz=1:size(A,3)].^3);
                                    @parallel_indices (ix,iy,iz) memopt=true loopsize=3 optvars=A optranges=(A=(0:0,0:0,0:0),) function copy_memopt!(A2, A)
                                        if ix>0 && ix<=size(A2,1) && iy>0 && iy<=size(A2,2) # TODO: needed when ranges is bigger than array
                                            A2[ix,iy,iz] = A[ix,iy,iz]
                                        end
                                        return
                                    end
                                    ranges = (1:64,1:64,1:8) # TODO: must be a multiple of the number of threads
                                    @parallel ranges memopt=true copy_memopt!(A2, A);
                                    @test all(Array(A2) .== Array(A))
                                end
                                @testset "@parallel <kernel> (3D, memopt, stencilranges=0:0)" begin
                                    A  = @Field(nxyz);
                                    A2 = @Field(nxyz);
                                    copy!(A, [ix + (iy-1)*size(A,1) + (iz-1)*size(A,1)*size(A,2) for ix=1:size(A,1), iy=1:size(A,2), iz=1:size(A,3)].^3);
                                    @parallel memopt=true loopsize=3 optvars=A optranges=(A=(0:0,0:0,0:0),) function copy_memopt!(A2, A)
                                        @all(A2) = @all(A)
                                        return
                                    end
                                    @parallel memopt=true copy_memopt!(A2, A);
                                    @test all(Array(A2) .== Array(A))
                                end
                                @testset "@parallel <kernel> (3D, memopt, stencilranges=0:2)" begin
                                    lam=dt=_dx=_dy=_dz = $FloatDefault(1)
                                    T      = @Field(nxyz);
                                    T2     = @Field(nxyz);
                                    T2_ref = @Field(nxyz);
                                    Ci     = @Field(nxyz, @ones);
                                    copy!(T, [ix + (iy-1)*size(T,1) + (iz-1)*size(T,1)*size(T,2) for ix=1:size(T,1), iy=1:size(T,2), iz=1:size(T,3)].^3);
                                    @parallel memopt=true loopsize=3 function diffusion3D_step!(T2, T, Ci, lam, dt, _dx, _dy, _dz)
                                        @inn(T2) = @inn(T) + dt*(lam*@inn(Ci)*(@d2_xi(T)*_dx^2 + @d2_yi(T)*_dy^2 + @d2_zi(T)*_dz^2))
                                        return
                                    end
                                    @parallel memopt=true diffusion3D_step!(T2, T, Ci, lam, dt, _dx, _dy, _dz);
                                    T2_ref[2:end-1,2:end-1,2:end-1] .= T[2:end-1,2:end-1,2:end-1] .+ dt.*(lam.*Ci[2:end-1,2:end-1,2:end-1].*(
                                                            ((T[3:end  ,2:end-1,2:end-1] .- T[2:end-1,2:end-1,2:end-1]) .- (T[2:end-1,2:end-1,2:end-1] .- T[1:end-2,2:end-1,2:end-1])).*_dx^2
                                                            + ((T[2:end-1,3:end  ,2:end-1] .- T[2:end-1,2:end-1,2:end-1]) .- (T[2:end-1,2:end-1,2:end-1] .- T[2:end-1,1:end-2,2:end-1])).*_dy^2
                                                            + ((T[2:end-1,2:end-1,3:end  ] .- T[2:end-1,2:end-1,2:end-1]) .- (T[2:end-1,2:end-1,2:end-1] .- T[2:end-1,2:end-1,1:end-2])).*_dz^2)
                                                            );
                                    @test all(Array(T2) .== Array(T2_ref))
                                end
                                @testset "@parallel <kernel> (3D, memopt; 3 arrays, x-y-z- + y- + x-stencil)" begin
                                    lam=dt=_dx=_dy=_dz = $FloatDefault(1)
                                    T      = @Field(nxyz);
                                    T2     = @Field(nxyz);
                                    T2_ref = @Field(nxyz);
                                    Ci     = @XYYZZField(nxyz);
                                    B      = @Field(nxyz);
                                    copy!(T, [ix + (iy-1)*size(T,1) + (iz-1)*size(T,1)*size(T,2) for ix=1:size(T,1), iy=1:size(T,2), iz=1:size(T,3)].^3);
                                    copy!(Ci, 2 .* [ix + (iy-1)*size(Ci,1) + (iz-1)*size(Ci,1)*size(Ci,2) for ix=1:size(Ci,1), iy=1:size(Ci,2), iz=1:size(Ci,3)].^3);
                                    copy!(B,  3 .* [ix + (iy-1)*size(B,1) + (iz-1)*size(B,1)*size(B,2) for ix=1:size(B,1), iy=1:size(B,2), iz=1:size(B,3)].^3);
                                    @parallel memopt=true loopsize=3 function diffusion3D_step_modified!(T2, T, Ci, B, lam, dt, _dx, _dy, _dz)
                                        @inn(T2) = @inn(T) + dt*(lam*@d_xi(Ci)*(@d2_xi(T)*_dx^2 + @d2_yi(T)*_dy^2 + @d2_zi(T)*_dz^2)) + @d2_yi(B)
                                        return
                                    end
                                    @parallel memopt=true diffusion3D_step_modified!(T2, T, Ci, B, lam, dt, _dx, _dy, _dz);
                                    T2_ref[2:end-1,2:end-1,2:end-1] .= T[2:end-1,2:end-1,2:end-1] .+ dt.*(lam.*(Ci[2:end,2:end-1,2:end-1] .- Ci[1:end-1,2:end-1,2:end-1]).*(
                                                            ((T[3:end  ,2:end-1,2:end-1] .- T[2:end-1,2:end-1,2:end-1]) .- (T[2:end-1,2:end-1,2:end-1] .- T[1:end-2,2:end-1,2:end-1])).*_dx^2
                                                            + ((T[2:end-1,3:end  ,2:end-1] .- T[2:end-1,2:end-1,2:end-1]) .- (T[2:end-1,2:end-1,2:end-1] .- T[2:end-1,1:end-2,2:end-1])).*_dy^2
                                                            + ((T[2:end-1,2:end-1,3:end  ] .- T[2:end-1,2:end-1,2:end-1]) .- (T[2:end-1,2:end-1,2:end-1] .- T[2:end-1,2:end-1,1:end-2])).*_dz^2)
                                                            ) + ((B[2:end-1,3:end  ,2:end-1] .- B[2:end-1,2:end-1,2:end-1]) .- (B[2:end-1,2:end-1,2:end-1] .- B[2:end-1,1:end-2,2:end-1]));
                                    @test all(Array(T2) .== Array(T2_ref))
                                end
                            end
                        end
                    )))
                end                
            end;
            @testset "@within" begin
                @test @prettystring(@within("@all", A)) == string(:(firstindex(A, 1) <= $ix  <= lastindex(A, 1) && (firstindex(A, 2) <= $iy  <= lastindex(A, 2) && firstindex(A, 3) <= $iz  <= lastindex(A, 3))))
                @test @prettystring(@within("@inn", A)) == string(:(firstindex(A, 1) <  $ixi <  lastindex(A, 1) && (firstindex(A, 2) <  $iyi <  lastindex(A, 2) && firstindex(A, 3) <  $izi <  lastindex(A, 3))))
            end;
            @testset "apply masks | handling padding (padding=false (default))" begin
                expansion = @prettystring(1, @parallel sum!(A, B) = (@all(A) = @all(A) + @all(B); return))
                @test occursin("if $ix_s <= size(A, 1) && ($iy_s <= size(A, 2) && $iz_s <= size(A, 3))", expansion)
                expansion = @prettystring(1, @parallel sum!(A, B) = (@inn(A) = @inn(A) + @inn(B); return))
                @test occursin("if $ix_s < size(A, 1) - 1 && ($iy_s < size(A, 2) - 1 && $iz_s < size(A, 3) - 1)", expansion)
                @test occursin("A[$ix_s + 1, $iy_s + 1, $iz_s + 1] = A[$ix_s + 1, $iy_s + 1, $iz_s + 1] + B[$ix_s + 1, $iy_s + 1, $iz_s + 1]", expansion)
            end;
            @testset "apply masks | handling padding (padding=true)" begin
                expansion = @prettystring(1, @parallel padding=true sum!(A, B) = (@all(A) = @all(A) + @all(B); return))
                @test occursin("if (A.indices[1])[1] <= $ix_s <= (A.indices[1])[end] && ((A.indices[2])[1] <= $iy_s <= (A.indices[2])[end] && (A.indices[3])[1] <= $iz_s <= (A.indices[3])[end])", expansion)
                expansion = @prettystring(1, @parallel padding=true sum!(A, B) = (@inn(A) = @inn(A) + @inn(B); return))
                @test occursin("if (A.indices[1])[1] < $ix_s < (A.indices[1])[end] && ((A.indices[2])[1] < $iy_s < (A.indices[2])[end] && (A.indices[3])[1] < $iz_s < (A.indices[3])[end])", expansion)
                @test occursin("A.parent[$ix_s, $iy_s, $iz_s] = A.parent[$ix_s, $iy_s, $iz_s] + B.parent[$ix_s, $iy_s, $iz_s]", expansion)
            end;
            @reset_parallel_stencil()
        end;
        @testset "2. parallel macros (2D)" begin
            @require !@is_initialized()
            @init_parallel_stencil($package, $FloatDefault, 2, nonconst_metadata=true)
            @require @is_initialized()
            @static if $package in [$PKG_CUDA, $PKG_AMDGPU] # TODO add support for Metal
                    nxy = (32, 8)
                    @testset "@parallel_indices <kernel> (2D, memopt, stencilranges=(0:0,-1:1,0:0))" begin
                        A      = @zeros(nxy...)
                        A2     = @zeros(nxy...)
                        A2_ref = @zeros(nxy...)
                        copy!(A, [ix + (iy - 1) * size(A, 1) for ix=1:size(A, 1), iy=1:size(A, 2)].^3)
                        @parallel_indices (ix, iy) ndims=2 memopt=true loopdim=2 function d2_memopt_indices!(A2, A)
                            if (1 < iy < size(A2, 2))
                                A2[ix, iy] = A[ix, iy + 1] - 2 * A[ix, iy] + A[ix, iy - 1]
                            end
                            return
                        end
                        @parallel d2_memopt_indices!(A2, A)
                        A2_ref[:, 2:end-1] .= A[:, 3:end] .- 2 * A[:, 2:end-1] .+ A[:, 1:end-2]
                        @test all(Array(A2) .== Array(A2_ref))
                    end;
                    @testset "@parallel <kernel> (2D, memopt, stencilranges=(0:0,-1:1,0:0))" begin
                        A2     = @zeros(nxy...)
                        B      = @zeros(nxy...)
                        A2_ref = @zeros(nxy...)
                        copy!(B, [ix + (iy - 1) * size(B, 1) for ix=1:size(B, 1), iy=1:size(B, 2)].^3)
                        @parallel ndims=2 memopt=true optvars=B loopdim=2 function d2_memopt_kernel!(A2, B)
                            @inn(A2) = ParallelStencil.FiniteDifferences2D.@d2_yi(B)
                            return
                        end
                        @parallel d2_memopt_kernel!(A2, B)
                        A2_ref[2:end-1, 2:end-1] .= B[2:end-1, 3:end] .- 2 * B[2:end-1, 2:end-1] .+ B[2:end-1, 1:end-2]
                        @test all(Array(A2) .== Array(A2_ref))
                    end;
                    @testset "@parallel_indices <kernel> (2D, memopt, optvars=A)" begin
                        A      = @zeros(nxy...)
                        B      = @zeros(nxy...)
                        A2     = @zeros(nxy...)
                        A2_ref = @zeros(nxy...)
                        copy!(A, [ix + (iy - 1) * size(A, 1) for ix=1:size(A, 1), iy=1:size(A, 2)].^3)
                        copy!(B, [2 * ix - iy for ix=1:size(B, 1), iy=1:size(B, 2)])
                        @parallel_indices (ix, iy) ndims=2 memopt=true loopdim=2 optvars=A function mixed_memopt_indices!(A2, A, B)
                            if (1 < ix < size(A2, 1) && 1 < iy < size(A2, 2))
                                A2[ix, iy] = (A[ix, iy + 1] - 2 * A[ix, iy] + A[ix, iy - 1]) + (B[ix + 1, iy] - 2 * B[ix, iy] + B[ix - 1, iy])
                            end
                            return
                        end
                        @parallel mixed_memopt_indices!(A2, A, B)
                        A2_ref[2:end-1, 2:end-1] .= (A[2:end-1, 3:end] .- 2 * A[2:end-1, 2:end-1] .+ A[2:end-1, 1:end-2]) .+ (B[3:end, 2:end-1] .- 2 * B[2:end-1, 2:end-1] .+ B[1:end-2, 2:end-1])
                        @test all(Array(A2) .== Array(A2_ref))
                    end;
                    @testset "@parallel_indices <kernel> (2D, memopt, optranges=(A=(0:0,-1:1,0:0),))" begin
                        A      = @zeros(nxy...)
                        A2     = @zeros(nxy...)
                        A2_ref = @zeros(nxy...)
                        copy!(A, [ix + (iy - 1) * size(A, 1) for ix=1:size(A, 1), iy=1:size(A, 2)].^3)
                        @parallel_indices (ix, iy) ndims=2 memopt=true loopdim=2 optvars=A optranges=(A=(0:0, -1:1, 0:0),) function ranged_memopt_indices!(A2, A)
                            if (1 < iy < size(A2, 2))
                                A2[ix, iy] = A[ix, iy + 1] - 2 * A[ix, iy] + A[ix, iy - 1]
                            end
                            return
                        end
                        @parallel ranged_memopt_indices!(A2, A)
                        A2_ref[:, 2:end-1] .= A[:, 3:end] .- 2 * A[:, 2:end-1] .+ A[:, 1:end-2]
                        @test all(Array(A2) .== Array(A2_ref))
                    end;
                    @testset "@parallel_indices <kernel> (2D, memopt, loopsize=4)" begin
                        A      = @zeros(nxy...)
                        A2     = @zeros(nxy...)
                        A2_ref = @zeros(nxy...)
                        copy!(A, [ix + (iy - 1) * size(A, 1) for ix=1:size(A, 1), iy=1:size(A, 2)].^3)
                        @parallel_indices (ix, iy) ndims=2 memopt=true loopdim=2 loopsize=4 function loopsize_memopt_indices!(A2, A)
                            if (1 < iy < size(A2, 2))
                                A2[ix, iy] = A[ix, iy + 1] - 2 * A[ix, iy] + A[ix, iy - 1]
                            end
                            return
                        end
                        @parallel loopsize_memopt_indices!(A2, A)
                        A2_ref[:, 2:end-1] .= A[:, 3:end] .- 2 * A[:, 2:end-1] .+ A[:, 1:end-2]
                        @test all(Array(A2) .== Array(A2_ref))
                    end;
                    @testset "@parallel_indices <kernel> (2D, memopt, optimize_halo_read=false)" begin
                        A      = @zeros(nxy...)
                        A2     = @zeros(nxy...)
                        A2_ref = @zeros(nxy...)
                        copy!(A, [ix + (iy - 1) * size(A, 1) for ix=1:size(A, 1), iy=1:size(A, 2)].^3)
                        @parallel_indices (ix, iy) ndims=2 memopt=true loopdim=2 optvars=A optranges=(A=(0:0, -1:1, 0:0),) optimize_halo_read=false function halo_read_memopt_indices!(A2, A)
                            if (1 < iy < size(A2, 2))
                                A2[ix, iy] = A[ix, iy + 1] - 2 * A[ix, iy] + A[ix, iy - 1]
                            end
                            return
                        end
                        @parallel halo_read_memopt_indices!(A2, A)
                        A2_ref[:, 2:end-1] .= A[:, 3:end] .- 2 * A[:, 2:end-1] .+ A[:, 1:end-2]
                        @test all(Array(A2) .== Array(A2_ref))
                    end;
            end;
            @reset_parallel_stencil()
        end;
        @testset "3. parallel <kernel> (with Fields)" begin
            @static if $package != $PKG_POLYESTER # TODO: this needs to be removed once Polyester supports padding
                @require !@is_initialized()
                @init_parallel_stencil($package, $FloatDefault, 3, padding=true, nonconst_metadata=true)
                @require @is_initialized()
                @testset "padding" begin
                    @testset "@parallel <kernel> (3D, @all)" begin
                        A = @Field((4, 5, 6));
                        @parallel function write_indices!(A)
                            @all(A) = $ix + ($iy-1)*size(A,1) + ($iz-1)*size(A,1)*size(A,2); # NOTE: $ix, $iy, $iz come from ParallelStencil.INDICES.
                            return
                        end
                        @parallel write_indices!(A);
                        @test all(Array(A) .== [ix + (iy-1)*size(A,1) + (iz-1)*size(A,1)*size(A,2) for ix=1:size(A,1), iy=1:size(A,2), iz=1:size(A,3)])
                    end
                    @testset "@parallel <kernel> (3D, @inn)" begin
                        A = @Field((4, 5, 6));
                        @parallel function write_indices!(A)
                            @inn(A) = $ixi + ($iyi-1)*size(A,1) + ($izi-1)*size(A,1)*size(A,2); # NOTE: $ix, $iy, $iz come from ParallelStencil.INDICES.
                            return
                        end
                        @parallel write_indices!(A);
                        @test all(Array(A)[2:end-1,2:end-1,2:end-1] .== ([ix + (iy-1)*size(A,1) + (iz-1)*size(A,1)*size(A,2) for ix=1:size(A,1), iy=1:size(A,2), iz=1:size(A,3)])[2:end-1,2:end-1,2:end-1])
                    end
                    @testset "@parallel <kernel> (3D; on-the-fly)" begin
                        nxyz   = (32, 8, 8)
                        lam=dt=_dx=_dy=_dz = $FloatDefault(1)
                        T      = @Field(nxyz);
                        T2     = @Field(nxyz);
                        T2_ref = @Field(nxyz);
                        Ci     = @Field(nxyz, @ones);
                        copy!(T, [ix + (iy-1)*size(T,1) + (iz-1)*size(T,1)*size(T,2) for ix=1:size(T,1), iy=1:size(T,2), iz=1:size(T,3)].^3);
                        @parallel function diffusion3D_step!(T2, T, Ci, lam::Data.Number, dt::$FloatDefault, _dx, _dy, _dz)
                            @all(qx)   = -lam*@d_xi(T)*_dx                                          # Fourier's law of heat conduction
                            @all(qy)   = -lam*@d_yi(T)*_dy                                          # ...
                            @all(qz)   = -lam*@d_zi(T)*_dz                                          # ...
                            @all(dTdt) = @inn(Ci)*(-@d_xa(qx)*_dx - @d_ya(qy)*_dy - @d_za(qz)*_dz)  # Conservation of energy
                            @inn(T2)   = @inn(T) + dt*@all(dTdt)                                    # Update of temperature
                            return
                        end
                        @parallel diffusion3D_step!(T2, T, Ci, lam, dt, _dx, _dy, _dz);
                        T2_ref[2:end-1,2:end-1,2:end-1] .= T[2:end-1,2:end-1,2:end-1] .+ dt.*(lam.*Ci[2:end-1,2:end-1,2:end-1].*(
                                                ((T[3:end  ,2:end-1,2:end-1] .- T[2:end-1,2:end-1,2:end-1]) .- (T[2:end-1,2:end-1,2:end-1] .- T[1:end-2,2:end-1,2:end-1])).*_dx^2
                                                + ((T[2:end-1,3:end  ,2:end-1] .- T[2:end-1,2:end-1,2:end-1]) .- (T[2:end-1,2:end-1,2:end-1] .- T[2:end-1,1:end-2,2:end-1])).*_dy^2
                                                + ((T[2:end-1,2:end-1,3:end  ] .- T[2:end-1,2:end-1,2:end-1]) .- (T[2:end-1,2:end-1,2:end-1] .- T[2:end-1,2:end-1,1:end-2])).*_dz^2)
                                                );
                        @test all(Array(T2) .== Array(T2_ref))
                    end
                end;
                @reset_parallel_stencil()
            end
        end;
        @testset "4. global defaults" begin
            @testset "inbounds=true" begin
                @require !@is_initialized()
                @init_parallel_stencil($package, $FloatDefault, 1, inbounds=true, nonconst_metadata=true)
                @require @is_initialized
                expansion = @prettystring(1, @parallel_indices (ix) inbounds=true f(A) = (2*A; return))
                @test occursin("Base.@inbounds begin", expansion)
                expansion = @prettystring(1, @parallel_indices (ix) f(A) = (2*A; return))
                @test occursin("Base.@inbounds begin", expansion)
                expansion = @prettystring(1, @parallel_indices (ix) inbounds=false f(A) = (2*A; return))
                @test !occursin("Base.@inbounds begin", expansion)
                @reset_parallel_stencil()
            end;
            @testset "padding=true" begin
                @static if $package != $PKG_POLYESTER # TODO: this needs to be removed once Polyester supports padding
                    @require !@is_initialized()
                    @init_parallel_stencil($package, $FloatDefault, 3, padding=true, nonconst_metadata=true)
                    @require @is_initialized
                    @testset "apply masks | handling padding (padding=true (globally))" begin
                        expansion = @prettystring(1, @parallel sum!(A, B) = (@all(A) = @all(A) + @all(B); return))
                        @test occursin("if (A.indices[1])[1] <= $ix_s <= (A.indices[1])[end] && ((A.indices[2])[1] <= $iy_s <= (A.indices[2])[end] && (A.indices[3])[1] <= $iz_s <= (A.indices[3])[end])", expansion)
                        expansion = @prettystring(1, @parallel sum!(A, B) = (@inn(A) = @inn(A) + @inn(B); return))
                        @test occursin("if (A.indices[1])[1] < $ix_s < (A.indices[1])[end] && ((A.indices[2])[1] < $iy_s < (A.indices[2])[end] && (A.indices[3])[1] < $iz_s < (A.indices[3])[end])", expansion)
                        @test occursin("A.parent[$ix_s, $iy_s, $iz_s] = A.parent[$ix_s, $iy_s, $iz_s] + B.parent[$ix_s, $iy_s, $iz_s]", expansion)
                    end;
                    @testset "apply masks | handling padding (padding=false)" begin
                        expansion = @prettystring(1, @parallel padding=false sum!(A, B) = (@all(A) = @all(A) + @all(B); return))
                        @test occursin("if $ix_s <= size(A, 1) && ($iy_s <= size(A, 2) && $iz_s <= size(A, 3))", expansion)
                        expansion = @prettystring(1, @parallel padding=false sum!(A, B) = (@inn(A) = @inn(A) + @inn(B); return))
                        @test occursin("if $ix_s < size(A, 1) - 1 && ($iy_s < size(A, 2) - 1 && $iz_s < size(A, 3) - 1)", expansion)
                        @test occursin("A[$ix_s + 1, $iy_s + 1, $iz_s + 1] = A[$ix_s + 1, $iy_s + 1, $iz_s + 1] + B[$ix_s + 1, $iy_s + 1, $iz_s + 1]", expansion)
                    end;
                    @reset_parallel_stencil()
                end
            end;
            @testset "@parallel_indices (I...) (1D)" begin
                @require !@is_initialized()
                @init_parallel_stencil($package, $FloatDefault, 1, nonconst_metadata=true)
                @require @is_initialized
                A  = @zeros(4*5*6)
                one = $FloatDefault(1)
                @parallel_indices (I...) function write_indices!(A, one)
                    A[I...] = sum((I .- (1,)) .* (one));
                    return
                end
                @parallel write_indices!(A, one);
                @test all(Array(A) .== [(ix-1) for ix=1:size(A,1)])
                @reset_parallel_stencil()
            end;
            @testset "@parallel_indices (I...) (2D)" begin
                @require !@is_initialized()
                @init_parallel_stencil($package, $FloatDefault, 2, nonconst_metadata=true)
                @require @is_initialized
                A  = @zeros(4, 5*6)
                one = $FloatDefault(1)
                @parallel_indices (I...) function write_indices!(A, one)
                    A[I...] = sum((I .- (1,)) .* (one, size(A,1)));
                    return
                end
                @parallel write_indices!(A, one);
                @test all(Array(A) .== [(ix-1) + (iy-1)*size(A,1) for ix=1:size(A,1), iy=1:size(A,2)])
                @reset_parallel_stencil()
            end;
            @testset "@parallel_indices (I...) (3D)" begin
                @require !@is_initialized()
                @init_parallel_stencil($package, $FloatDefault, 3, nonconst_metadata=true)
                @require @is_initialized
                A  = @zeros(4, 5, 6)
                one = $FloatDefault(1)
                @parallel_indices (I...) function write_indices!(A, one)
                    A[I...] = sum((I .- (1,)) .* (one, size(A,1), size(A,1)*size(A,2)));
                    return
                end
                @parallel write_indices!(A, one);
                @test all(Array(A) .== [(ix-1) + (iy-1)*size(A,1) + (iz-1)*size(A,1)*size(A,2) for ix=1:size(A,1), iy=1:size(A,2), iz=1:size(A,3)])
                @reset_parallel_stencil()
            end;
        end;
        @testset "5. parallel macros (numbertype and ndims ommited)" begin
            @require !@is_initialized()
            @init_parallel_stencil(package = $package, nonconst_metadata=true)
            @require @is_initialized
            $(interpolate(:__T__, ARRAYTYPES, :(
                @testset "Data.__T__{T} to Data.Device.__T__{T}" begin
                    @static if @isgpu($package)
                        expansion = @prettystring(1, @parallel ndims=3 f(A::Data.__T__{T}, B::Data.__T__{T}, c::Integer) where T <: PSNumber = (@all(A) = @all(B)^c; return))
                        @test occursin("f(A::Data.Device.__T__{T}, B::Data.Device.__T__{T},", expansion)
                    end
                end;
            )))
            $(interpolate(:__T__, FIELDTYPES, :(
                @testset "Data.Fields.__T__{T} to Data.Fields.Device.__T__{T}" begin
                    @static if @isgpu($package)
                        expansion = @prettystring(1, @parallel ndims=3 f(A::Data.Fields.__T__{T}, B::Data.Fields.__T__{T}, c::Integer) where T <: PSNumber = (@all(A) = @all(B)^c; return))
                        @test occursin("f(A::Data.Fields.Device.__T__{T}, B::Data.Fields.Device.__T__{T},", expansion)
                    end
                end;
            )))
            @testset "N substitution | ndims tuple expansion" begin
                @testset "@parallel" begin
                    @testset "N substitution (ndims=2, N=3)" begin
                        expansion = @prettystring(1, @parallel ndims=2 N=3 f(A::Data.Array{T,N}, B::Data.Array{T,N}, c::Integer) where T <: PSNumber = (@all(A) = @all(B)^c; return))
                        @test occursin("f(A::Data.Array{T, 3}, B::Data.Array{T, 3},", expansion)
                    end;
                    @testset "N substitution (ndims=2, N=2)" begin
                        expansion = @prettystring(1, @parallel ndims=2 N=2 f(A::Data.Array{T,N}, B::Data.Array{T,N}, c::Integer) where T <: PSNumber = (@all(A) = @all(B)^c; return))
                        @test occursin("f(A::Data.Array{T, 2}, B::Data.Array{T, 2},", expansion)
                    end;
                    @testset "N substitution (ndims=2, N=1)" begin
                        expansion = @prettystring(1, @parallel ndims=2 N=1 f(A::Data.Array{T,N}, B::Data.Array{T,N}, c::Integer) where T <: PSNumber = (@all(A) = @all(B)^c; return))
                        @test occursin("f(A::Data.Array{T, 1}, B::Data.Array{T, 1},", expansion)
                    end;
                    @testset "N substitution (ndims=2, N=ndims)" begin
                        expansion = @prettystring(1, @parallel ndims=2 N=ndims f(A::Data.Array{T,N}, B::Data.Array{T,N}, c::Integer) where T <: PSNumber = (@all(A) = @all(B)^c; return))
                        @test occursin("f(A::Data.Array{T, 2}, B::Data.Array{T, 2},", expansion)
                    end;
                    @testset "N substitution (ndims=2, N=ndims+2)" begin
                        expansion = @prettystring(1, @parallel ndims=2 N=ndims+2 f(A::Data.Array{T,N}, B::Data.Array{T,N}, c::Integer) where T <: PSNumber = (@all(A) = @all(B)^c; return))
                        @test occursin("f(A::Data.Array{T, 4}, B::Data.Array{T, 4},", expansion)
                    end;
                    @testset "ndims tuple expansion (ndims=(1,2,3), N=ndims)" begin
                        expansion = @prettystring(2, @parallel ndims=(1,2,3) N=ndims f(A::Data.Array{T,N}, B::Data.Array{T,N}, c::Integer) where T <: PSNumber = (@all(A) = @all(B)^c; return))
                        @test occursin("@parallel ndims = 1 function f(A::Data.Array{T, 1}, B::Data.Array{T, 1},", expansion)
                        @test occursin("@parallel ndims = 2 function f(A::Data.Array{T, 2}, B::Data.Array{T, 2},", expansion)
                        @test occursin("@parallel ndims = 3 function f(A::Data.Array{T, 3}, B::Data.Array{T, 3},", expansion)
                    end;
                    @testset "ndims tuple expansion (ndims=(1,3), N=ndims.+1)" begin
                        expansion = @prettystring(2, @parallel ndims=(1,3) N=ndims.+1 f(A::Data.Array{T,N}, B::Data.Array{T,N}, c::Integer) where T <: PSNumber = (@all(A) = @all(B)^c; return))
                        @test occursin("@parallel ndims = 1 function f(A::Data.Array{T, 2}, B::Data.Array{T, 2},", expansion)
                        @test occursin("@parallel ndims = 3 function f(A::Data.Array{T, 4}, B::Data.Array{T, 4},", expansion)
                        @test !occursin("@parallel ndims = 2", expansion)
                    end;
                end;
                @testset "@parallel_indices" begin
                    @testset "N substitution (ndims=2, N=3)" begin
                        expansion = @prettystring(1, @parallel_indices I... ndims=2 N=3 f(A::Data.Array{T,N}, B::Data.Array{T,N}, c::Integer) where T <: PSNumber = (A[I...] = B[I...]^c; return))
                        @test occursin("f(A::Data.Array{T, 3}, B::Data.Array{T, 3},", expansion)
                    end;
                    @testset "N substitution (ndims=2, N=2)" begin
                        expansion = @prettystring(1, @parallel_indices I... ndims=2 N=2 f(A::Data.Array{T,N}, B::Data.Array{T,N}, c::Integer) where T <: PSNumber = (A[I...] = B[I...]^c; return))
                        @test occursin("f(A::Data.Array{T, 2}, B::Data.Array{T, 2},", expansion)
                    end;
                    @testset "N substitution (ndims=2, N=1)" begin
                        expansion = @prettystring(1, @parallel_indices I... ndims=2 N=1 f(A::Data.Array{T,N}, B::Data.Array{T,N}, c::Integer) where T <: PSNumber = (A[I...] = B[I...]^c; return))
                        @test occursin("f(A::Data.Array{T, 1}, B::Data.Array{T, 1},", expansion)
                    end;
                    @testset "N substitution (ndims=2, N=ndims)" begin
                        expansion = @prettystring(1, @parallel_indices I... ndims=2 N=ndims f(A::Data.Array{T,N}, B::Data.Array{T,N}, c::Integer) where T <: PSNumber = (A[I...] = B[I...]^c; return))
                        @test occursin("f(A::Data.Array{T, 2}, B::Data.Array{T, 2},", expansion)
                    end;
                    @testset "N substitution (ndims=2, N=ndims+2)" begin
                        expansion = @prettystring(1, @parallel_indices I... ndims=2 N=ndims+2 f(A::Data.Array{T,N}, B::Data.Array{T,N}, c::Integer) where T <: PSNumber = (A[I...] = B[I...]^c; return))
                        @test occursin("f(A::Data.Array{T, 4}, B::Data.Array{T, 4},", expansion)
                    end;
                    @testset "ndims tuple expansion (ndims=(1,2,3), N=ndims)" begin
                        expansion = @prettystring(2, @parallel_indices I... ndims=(1,2,3) N=ndims f(A::Data.Array{T,N}, B::Data.Array{T,N}, c::Integer) where T <: PSNumber = (A[I...] = B[I...]^c; return))
                        @test occursin("@parallel_indices I... ndims = 1 function f(A::Data.Array{T, 1}, B::Data.Array{T, 1},", expansion)
                        @test occursin("@parallel_indices I... ndims = 2 function f(A::Data.Array{T, 2}, B::Data.Array{T, 2},", expansion)
                        @test occursin("@parallel_indices I... ndims = 3 function f(A::Data.Array{T, 3}, B::Data.Array{T, 3},", expansion)
                    end;
                    @testset "ndims tuple expansion (ndims=(1,3), N=ndims.+1)" begin
                        expansion = @prettystring(2, @parallel_indices I... ndims=(1,3) N=ndims.+1 f(A::Data.Array{T,N}, B::Data.Array{T,N}, c::Integer) where T <: PSNumber = (A[I...] = B[I...]^c; return))
                        @test occursin("@parallel_indices I... ndims = 1 function f(A::Data.Array{T, 2}, B::Data.Array{T, 2},", expansion)
                        @test occursin("@parallel_indices I... ndims = 3 function f(A::Data.Array{T, 4}, B::Data.Array{T, 4},", expansion)
                        @test !occursin("@parallel_indices I... ndims = 2", expansion)
                    end;
                end;
            end;
            @reset_parallel_stencil()
        end;
        @testset "6. metadata" begin
            @require !@is_initialized()
            @init_parallel_stencil($package, $FloatDefault, 3, nonconst_metadata=true)
            @require @is_initialized()
            @testset "standard" begin
                @parallel_indices (ix, iy, iz) function metadata_probe!(A, B, D)
                    A[ix, iy, iz] = 2.0 * B[ix, iy, iz]
                    return
                end
                A = @zeros(4, 5, 6)
                B = @ones(4, 5, 6)
                D = @ones(4, 5, 6, 2)
                metadata = @metadata metadata_probe!(A, B, D)
                metadata_symbols = sort(setdiff(names(metadata; all=true), names(metadata)))
                @test metadata isa Module
                @test length(names(metadata)) == 1
                @test metadata_symbols == [:memopt, :nb_parallel_indices]
                @test metadata.memopt == false
                @test metadata.nb_parallel_indices == 3
                @test all(Array(A) .== 0)
            end;
            @static if $package != $PKG_KERNELABSTRACTIONS
                @testset "memopt" begin
                    @parallel_indices (ix, iy, iz) memopt=true loopsize=3 optvars=B optranges=(B=(0:0,0:0,0:0),) function metadata_memopt_probe!(A, B, D)
                        A[ix, iy, iz] = 2.0 * B[ix, iy, iz]
                        return
                    end
                    A = @zeros(4, 5, 6)
                    B = @ones(4, 5, 6)
                    D = @ones(4, 5, 6, 2)
                    metadata = @metadata metadata_memopt_probe!(A, B, D)
                    metadata_symbols = sort(setdiff(names(metadata; all=true), names(metadata)))
                    @test metadata isa Module
                    @test length(names(metadata)) == 1
                    @test metadata_symbols == [:is_parallel_kernel, :loopdim, :loopsize, :loopsizes, :memopt, :nb_parallel_indices, :nonconst_metadata, :offsets, :optranges, :optvars, :shmem_dim1, :shmem_dim2, :shmem_optvars, :shmem_spans, :stencilranges, :use_any_shmem, :use_shmemhalos]
                    @test metadata.is_parallel_kernel == false
                    @test metadata.loopdim == 3
                    @test metadata.loopsize == 3
                    @test metadata.loopsizes == (1, 1, 3)
                    @test metadata.memopt == true
                    @test metadata.nb_parallel_indices == 3
                    @test metadata.nonconst_metadata == true
                    @test metadata.offsets[:B][(0, 0)][0] == 1
                    @test metadata.optranges[:B] == (0:0, 0:0, 0:0)
                    @test metadata.optvars == (:B,)
                    @test metadata.shmem_dim1 == 1
                    @test metadata.shmem_dim2 == 2
                    @test metadata.shmem_optvars == ()
                    @test metadata.shmem_optvars isa NTuple{0,Symbol}
                    @test metadata.shmem_spans == (B = (0, 0),)
                    @test metadata.stencilranges == (B = (0:0, 0:0, 0:0),)
                    @test metadata.use_any_shmem == false
                    @test metadata.use_shmemhalos[:B] == true
                    @parallel_indices (ix, iy, iz) memopt=true loopsize=3 optvars=B optranges=(B=(0:0,0:0,-1:1),) function metadata_memopt_zstencil_probe!(A, B, D)
                        A[ix, iy, iz] = B[ix, iy, iz-1] + B[ix, iy, iz] + B[ix, iy, iz+1] + D[ix, iy, iz, 1]
                        return
                    end
                    metadata_z = @metadata metadata_memopt_zstencil_probe!(A, B, D)
                    @test metadata_z.shmem_optvars == ()
                    @test metadata_z.shmem_optvars isa NTuple{0,Symbol}
                    @test metadata_z.shmem_spans == (B = (0, 0),)
                    @test metadata_z.stencilranges == (B = (0:0, 0:0, -1:1),)
                    @test metadata_z.use_any_shmem == false
                    @parallel_indices (ix, iy, iz) memopt=true loopsize=3 optvars=B function metadata_memopt_fullstencil_probe!(A, B, D)
                        A[ix, iy, iz] = B[ix-1, iy-1, iz-1] + B[ix, iy, iz] + B[ix+1, iy+1, iz+1] + D[ix, iy, iz, 1]
                        return
                    end
                    metadata_full = @metadata metadata_memopt_fullstencil_probe!(A, B, D)
                    @static if @isgpu($package)
                        @test metadata_full.shmem_optvars == (:B,)
                        @test metadata_full.shmem_optvars isa NTuple{1,Symbol}
                        @test metadata_full.shmem_spans == (B = (2, 2),)
                        @test metadata_full.stencilranges == (B = (-1:1, -1:1, -1:1),)
                        @test metadata_full.use_any_shmem == true
                    else
                        @test metadata_full.shmem_optvars == ()
                        @test metadata_full.shmem_optvars isa NTuple{0,Symbol}
                        @test metadata_full.shmem_spans == (B = (0, 0),)
                        @test metadata_full.stencilranges == (B = (0:0, 0:0, 0:0),)
                        @test metadata_full.use_any_shmem == false
                    end
                    @parallel_indices (ix, iy) ndims=2 memopt=true loopdim=2 loopsize=5 optvars=B optranges=(B=(0:0,-1:1,0:0),) function metadata_memopt_2d_probe!(A, B)
                        if (1 < iy < size(A, 2))
                            A[ix, iy] = B[ix, iy - 1] + B[ix, iy] + B[ix, iy + 1]
                        end
                        return
                    end
                    A2 = @zeros(4, 5)
                    B2 = @ones(4, 5)
                    metadata_2d = @metadata metadata_memopt_2d_probe!(A2, B2)
                    @test metadata_2d.nb_parallel_indices == 2
                    @test metadata_2d.loopdim == 2
                    @test metadata_2d.loopsize == 5
                    @test metadata_2d.loopsizes == (1, 5, 1)
                    @test metadata_2d.memopt == true
                    @test metadata_2d.optranges[:B] == (0:0, -1:1, 0:0)
                    @test metadata_2d.optvars == (:B,)
                    @test metadata_2d.shmem_dim1 == 1
                    @test metadata_2d.shmem_dim2 == 3
                    @test metadata_2d.shmem_optvars == ()
                    @test metadata_2d.shmem_optvars isa NTuple{0,Symbol}
                    @test metadata_2d.shmem_spans == (B = (0, 0),)
                    @test metadata_2d.stencilranges == (B = (0:0, -1:1, 0:0),)
                    @test metadata_2d.use_any_shmem == false
                    @test metadata_2d.use_shmemhalos[:B] == false
                    @test ParallelStencil.compute_memopt_shmem(Val(metadata_2d.shmem_optvars), Val(metadata_2d.use_shmemhalos), Val(metadata_2d.shmem_spans), Val(metadata_2d.shmem_dim1), Val(metadata_2d.shmem_dim2), (8, 4, 1), $FloatDefault) == 0
                    @test all(Array(A) .== 0)
                end;
            end;
            @reset_parallel_stencil()
        end;
        @testset "7. Exceptions" begin
            @init_parallel_stencil($package, $FloatDefault, 3, nonconst_metadata=true)
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
            @testset "automatic ranges: error if not all parallel indices are used" begin
                @test_throws ArgumentError parallel_indices(:((ix, iy, iz)), 
                :(function write_xy_plane!(A)
                    A[ix, iy, 1] = 2.0 * A[ix, iy, 1]
                    return
                end))
                @test_throws ArgumentError parallel_indices(:((ix, iy)), 
                :(function write_y_line!(A)
                    A[1, iy] = 2.0 * A[1, iy]
                    return
                end))
            end;
            @testset "automatic ranges: error if input array has more dimensions than parallel indices" begin
                @parallel_indices (ix, iy) function write_xy_plane!(A)
                    A[ix, iy, 1] = 2.0 * A[ix, iy, 1]
                    A[ix, iy, 2] = 2.0 * A[ix, iy, 2]
                    return
                end
                @parallel_indices (ix) function write_x_line!(A)
                    A[ix, 1] = 2.0 * A[ix, 1]
                    A[ix, 2] = 2.0 * A[ix, 2]
                    return
                end
                A3 = @ones(4, 5, 2)
                A2 = @ones(4, 2)
                @test_throws ArgumentError @parallel write_xy_plane!(A3)
                @test_throws ArgumentError @parallel write_x_line!(A2)
                @parallel (1:size(A3,1), 1:size(A3,2)) write_xy_plane!(A3)
                @parallel (1:size(A2,1)) write_x_line!(A2)
                @test A3 == 2.0 .* @ones(4, 5, 2)
                @test A2 == 2.0 .* @ones(4, 2)
            end;
            @testset "automatic ranges (memopt): error if not all parallel indices are used" begin
                @test_throws ArgumentError parallel_indices(:((ix, iy, iz)), :(memopt=true), 
                :(function write_xy_plane!(A, B)
                    A[ix, iy, 1] = 2.0 * A[ix, iy, 1] + B[ix, iy, 1]
                    return
                end))
                @test_throws ArgumentError parallel_indices(:((ix, iy)), :(memopt=true), 
                :(function write_y_line!(A, B)
                    A[1, iy] = 2.0 * A[1, iy] + B[1, iy]
                    return
                end))
            end;
            @static if $package != $PKG_KERNELABSTRACTIONS
                @testset "automatic ranges (memopt): error if input array has more dimensions than parallel indices" begin
                    @parallel_indices (ix, iy, iz) memopt=true loopsize=3 optvars=B optranges=(B=(0:0,0:0,0:0),) function write_xy_plane!(A, B, D)
                        A[ix, iy, iz] = 2.0 * B[ix, iy, iz]
                        return
                    end
                    A3 = @zeros(4, 5, 6)
                    B3 = @ones(4, 5, 6)
                    D4 = @ones(4, 5, 6, 2)
                    @test_throws ArgumentError @parallel memopt=true write_xy_plane!(A3, B3, D4)
                    @static if $package in [$PKG_CUDA, $PKG_AMDGPU]
                        @parallel (1:size(A3,1), 1:size(A3,2), 1:size(A3,3)) memopt=true write_xy_plane!(A3, B3, D4)
                        @test A3 == 2.0 .* @ones(4, 5, 6)
                    end
                end;
                @testset "declaration (memopt): error if 2-D kernels use shared-memory keywords or unsupported dimensionalities" begin
                    @test_throws ArgumentError parallel_indices(:((ix, iy)), :(ndims=2), :(memopt=true), :(loopdim=2), :(useshmemhalos=(A=true,)),
                    :(function invalid_2d_memopt_indices!(A2, A)
                        if (1 < iy < size(A2, 2))
                            A2[ix, iy] = A[ix, iy + 1] - 2 * A[ix, iy] + A[ix, iy - 1]
                        end
                        return
                    end))
                    @test_throws ArgumentError parallel(LineNumberNode(@__LINE__, Symbol(@__FILE__)), @__MODULE__, :(ndims=2), :(memopt=true), :(optvars=B), :(loopdim=2), :(useshmemhalos=(B=true,)),
                    :(function invalid_2d_memopt_kernel!(A2, B)
                        @inn(A2) = ParallelStencil.FiniteDifferences2D.@d2_yi(B)
                        return
                    end))
                    @test_throws ArgumentError parallel_indices(:((ix, iy, iz, iw)), :(memopt=true), :(loopdim=4),
                    :(function invalid_4d_memopt_indices!(A2, A)
                        A2[ix, iy, iz, iw] = A[ix, iy, iz, iw]
                        return
                    end))
                end;
            end;
            @reset_parallel_stencil()
        end;
    end;
))

end == nothing || true;
