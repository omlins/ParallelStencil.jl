using Test
using ParallelStencil
import ParallelStencil: @reset_parallel_stencil, @is_initialized, SUPPORTED_PACKAGES,
    PKG_CUDA, PKG_AMDGPU, PKG_METAL, PKG_THREADS, PKG_POLYESTER, @require, @iscpu

TEST_PACKAGES = SUPPORTED_PACKAGES
@static if PKG_CUDA in TEST_PACKAGES
    import CUDA
    if !CUDA.functional()
        TEST_PACKAGES = filter!(x -> x ≠ PKG_CUDA, TEST_PACKAGES)
    end
end
@static if PKG_AMDGPU in TEST_PACKAGES
    import AMDGPU
    if !AMDGPU.functional()
        TEST_PACKAGES = filter!(x -> x ≠ PKG_AMDGPU, TEST_PACKAGES)
    end
end
@static if PKG_METAL in TEST_PACKAGES
    import Metal
    if !Metal.functional()
        TEST_PACKAGES = filter!(x -> x ≠ PKG_METAL, TEST_PACKAGES)
    end
end
@static if PKG_POLYESTER in TEST_PACKAGES
    import Polyester
end
Base.retry_load_extensions()

strip_linenums(ex) = Base.remove_linenums!(deepcopy(ex))
expand_once(expr) = strip_linenums(Base.macroexpand(@__MODULE__, expr, recursive=false))
normalized_string(expr) = strip(replace(string(expand_once(expr)), r"#= .*? =#" => ""))

@static for package in TEST_PACKAGES
    FloatDefault = (package == PKG_METAL) ? Float32 : Float64

    eval(:(
        @testset "$(basename(@__FILE__)) (package: $(nameof($package)))" begin
            @testset "Kernel language pass-through macros" begin
                @require !@is_initialized()
                @init_parallel_stencil($package, $FloatDefault, 3, nonconst_metadata=true)
                @require @is_initialized()

                mask64    = UInt64(0xffff_ffff_ffff_ffff)
                mask32    = UInt32(0xffff_ffff)
                val       = one($FloatDefault)
                lane      = 1
                width     = 2
                delta     = 1
                lanemask  = 1
                predicate = true
                x         = 42

                @testset "Macro expansion forwards to ParallelKernel" begin
                    @test normalized_string(:(@gridDim()))      == "ParallelStencil.ParallelKernel.@gridDim"
                    @test normalized_string(:(@blockIdx()))     == "ParallelStencil.ParallelKernel.@blockIdx"
                    @test normalized_string(:(@blockDim()))     == "ParallelStencil.ParallelKernel.@blockDim"
                    @test normalized_string(:(@threadIdx()))    == "ParallelStencil.ParallelKernel.@threadIdx"
                    @test normalized_string(:(@sync_threads())) == "ParallelStencil.ParallelKernel.@sync_threads"
                    @test normalized_string(:(@sharedMem($FloatDefault, (2, 3)))) == "ParallelStencil.ParallelKernel.@sharedMem $(string($FloatDefault)) (2, 3)"
                    @test normalized_string(:(@ps_show x))      == "ParallelStencil.ParallelKernel.@pk_show x"
                    @test normalized_string(:(@ps_println "pass-through")) == "ParallelStencil.ParallelKernel.@pk_println \"pass-through\""
                    @test occursin("ParallelStencil.ParallelKernel.@∀", normalized_string(:(@∀ i ∈ (x,) @all(C.i) = @all(A.i))))

                    @test normalized_string(:(@warpsize()))     == "ParallelStencil.ParallelKernel.@warpsize"
                    @test normalized_string(:(@laneid()))       == "ParallelStencil.ParallelKernel.@laneid"
                    @test normalized_string(:(@active_mask()))  == "ParallelStencil.ParallelKernel.@active_mask"

                    @test normalized_string(:(@shfl_sync(mask32, val, lane))) == "ParallelStencil.ParallelKernel.@shfl_sync mask32 val lane"
                    @test normalized_string(:(@shfl_sync(mask32, val, lane, width))) == "ParallelStencil.ParallelKernel.@shfl_sync mask32 val lane width"
                    @test normalized_string(:(@shfl_up_sync(mask32, val, delta))) == "ParallelStencil.ParallelKernel.@shfl_up_sync mask32 val delta"
                    @test normalized_string(:(@shfl_up_sync(mask32, val, delta, width))) == "ParallelStencil.ParallelKernel.@shfl_up_sync mask32 val delta width"
                    @test normalized_string(:(@shfl_down_sync(mask32, val, delta))) == "ParallelStencil.ParallelKernel.@shfl_down_sync mask32 val delta"
                    @test normalized_string(:(@shfl_down_sync(mask32, val, delta, width))) == "ParallelStencil.ParallelKernel.@shfl_down_sync mask32 val delta width"
                    @test normalized_string(:(@shfl_xor_sync(mask32, val, lanemask))) == "ParallelStencil.ParallelKernel.@shfl_xor_sync mask32 val lanemask"
                    @test normalized_string(:(@shfl_xor_sync(mask32, val, lanemask, width))) == "ParallelStencil.ParallelKernel.@shfl_xor_sync mask32 val lanemask width"

                    @test normalized_string(:(@vote_any_sync(mask32, predicate))) == "ParallelStencil.ParallelKernel.@vote_any_sync mask32 predicate"
                    @test normalized_string(:(@vote_all_sync(mask32, predicate))) == "ParallelStencil.ParallelKernel.@vote_all_sync mask32 predicate"
                    @test normalized_string(:(@vote_ballot_sync(mask32, predicate))) == "ParallelStencil.ParallelKernel.@vote_ballot_sync mask32 predicate"
                end

                @testset "CPU runtime smoke tests" begin
                    @static if @iscpu($package)
                        N = 8
                        A = rand($FloatDefault, N)
                        P = [A[i] > zero($FloatDefault) for i in 1:N]
                        Bout_any    = Vector{Bool}(undef, N)
                        Bout_all    = Vector{Bool}(undef, N)
                        Bout_ballot = Vector{UInt64}(undef, N)
                        Bshfl       = similar(A)
                        Bshfl_up    = similar(A)
                        Bshfl_down  = similar(A)
                        Bshfl_xor   = similar(A)

                        @parallel_indices (ix) function kernel_pass_through!(Bout_any, Bout_all, Bout_ballot,
                                Bshfl, Bshfl_up, Bshfl_down, Bshfl_xor, A, P)
                            mask = @active_mask()
                            warp = @warpsize()
                            lane_local = @laneid()
                            @test warp == 1
                            @test lane_local == 1

                            Bshfl[ix]      = @shfl_sync(mask, A[ix], lane_local)
                            Bshfl_up[ix]   = @shfl_up_sync(mask, A[ix], 1)
                            Bshfl_down[ix] = @shfl_down_sync(mask, A[ix], 1)
                            Bshfl_xor[ix]  = @shfl_xor_sync(mask, A[ix], 1)

                            pred = P[ix]
                            Bout_any[ix]    = @vote_any_sync(mask, pred)
                            Bout_all[ix]    = @vote_all_sync(mask, pred)
                            Bout_ballot[ix] = @vote_ballot_sync(mask, pred)
                            return
                        end

                        @parallel (1:N) kernel_pass_through!(Bout_any, Bout_all, Bout_ballot,
                            Bshfl, Bshfl_up, Bshfl_down, Bshfl_xor, A, P)

                        @test all(Bshfl .== A)
                        @test all(Bshfl_up .== A)
                        @test all(Bshfl_down .== A)
                        @test all(Bshfl_xor .== A)
                        @test Bout_any == P
                        @test Bout_all == P
                        @test Bout_ballot == map(p -> p ? UInt64(0x1) : UInt64(0x0), P)
                    end
                end

                @reset_parallel_stencil()
                @require !@is_initialized()
            end
        end
    ))
end == nothing || true;
