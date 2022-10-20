using Test
import ParallelStencil
using ParallelStencil.ParallelKernel
import ParallelStencil.ParallelKernel: @reset_parallel_kernel, @is_initialized, SUPPORTED_PACKAGES, PKG_CUDA, PKG_AMDGPU, PKG_THREADS
import ParallelStencil.ParallelKernel: @require, @prettystring
import ParallelStencil.ParallelKernel: checknoargs, checkargs_sharedMem, Dim3
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

@static for package in TEST_PACKAGES  eval(:(
    @testset "$(basename(@__FILE__)) (package: $(nameof($package)))" begin
        @testset "1. kernel language macros" begin
            @testset "mapping to package" begin
                @require !@is_initialized()
                @init_parallel_kernel($package, Float64)
                @require @is_initialized()
                @static if $(QuoteNode(package)) == $(QuoteNode(PKG_CUDA))
                    @test @prettystring(1, @gridDim()) == "CUDA.gridDim()"
                    @test @prettystring(1, @blockIdx()) == "CUDA.blockIdx()"
                    @test @prettystring(1, @blockDim()) == "CUDA.blockDim()"
                    @test @prettystring(1, @threadIdx()) == "CUDA.threadIdx()"
                    @test @prettystring(1, @sync_threads()) == "CUDA.sync_threads()"
                    @test @prettystring(1, @sharedMem(Float32, (2,3))) == "CUDA.@cuDynamicSharedMem Float32 (2, 3)"
                    @test @prettystring(1, @pk_show()) == "CUDA.@cushow"
                    @test @prettystring(1, @pk_println()) == "CUDA.@cuprintln"
                elseif $(QuoteNode(package)) == $(QuoteNode(AMDGPU))
                    @test @prettystring(1, @gridDim()) == "AMDGPU.gridDimWG()"
                    @test @prettystring(1, @blockIdx()) == "AMDGPU.workgroupIdx()"
                    @test @prettystring(1, @blockDim()) == "AMDGPU.workgroupDim()"
                    @test @prettystring(1, @threadIdx()) == "AMDGPU.workitemIdx()"
                    @test @prettystring(1, @sync_threads()) == "AMDGPU.sync_workgroup()"
                    #@test @prettystring(1, @sharedMem(Float32, (2,3))) == ""    #TODO: not yet supported for AMDGPU
                    # @test @prettystring(1, @pk_show()) == "CUDA.@cushow"        #TODO: not yet supported for AMDGPU
                    # @test @prettystring(1, @pk_println()) == "CUDA.@cuprintln"  #TODO: not yet supported for AMDGPU
                elseif $(QuoteNode(package)) == $(QuoteNode(PKG_THREADS))
                    @test @prettystring(1, @gridDim()) == "ParallelStencil.ParallelKernel.@gridDim_cpu"
                    @test @prettystring(1, @blockIdx()) == "ParallelStencil.ParallelKernel.@blockIdx_cpu"
                    @test @prettystring(1, @blockDim()) == "ParallelStencil.ParallelKernel.@blockDim_cpu"
                    @test @prettystring(1, @threadIdx()) == "ParallelStencil.ParallelKernel.@threadIdx_cpu"
                    @test @prettystring(1, @sync_threads()) == "ParallelStencil.ParallelKernel.@sync_threads_cpu"
                    @test @prettystring(1, @sharedMem(Float32, (2,3))) == "ParallelStencil.ParallelKernel.@sharedMem_cpu Float32 (2, 3)"
                    @test @prettystring(1, @pk_show()) == "Base.@show"
                    @test @prettystring(1, @pk_println()) == "Base.@println"
                end;
            end;
            @testset "@gridDim, @blockIdx, @blockDim, @threadIdx (1D)" begin
                @static if $(QuoteNode(package)) == $(QuoteNode(PKG_THREADS))
                    A  = @zeros(4)
                    @parallel_indices (ix) function test_macros!(A)
                        @test @gridDim() == Dim3(2, 1, 1)
                        @test @blockIdx() == Dim3(ix-2, 1, 1)
                        @test @blockDim() == Dim3(1, 1, 1)
                        @test @threadIdx() == Dim3(1, 1, 1)
                        return
                    end
                    @parallel (3:4) test_macros!(A);
                    @test true # @gridDim test succeeded if this line is reached (the above tests within the kernel are not captured if they succeed, only if they fail...; alternative test implementations would be of course possible, but would be more complex).
                    @test true # @blockIdx ...
                    @test true # @blockDim ...
                    @test true # @threadIdx ...
                end
            end;
            @testset "@gridDim, @blockIdx, @blockDim, @threadIdx (2D)" begin
                @static if $(QuoteNode(package)) == $(QuoteNode(PKG_THREADS))
                    A  = @zeros(4, 5)
                    @parallel_indices (ix,iy) function test_macros!(A)
                        @test @gridDim() == Dim3(2, 3, 1)
                        @test @blockIdx() == Dim3(ix-2, iy-1, 1)
                        @test @blockDim() == Dim3(1, 1, 1)
                        @test @threadIdx() == Dim3(1, 1, 1)
                        return
                    end
                    @parallel (3:4, 2:4) test_macros!(A);
                    @test true # @gridDim test succeeded if this line is reached (the above tests within the kernel are not captured if they succeed, only if they fail...; alternative test implementations would be of course possible, but would be more complex).
                    @test true # @blockIdx ...
                    @test true # @blockDim ...
                    @test true # @threadIdx ...
                end
            end;
            @testset "@gridDim, @blockIdx, @blockDim, @threadIdx (3D)" begin
                @static if $(QuoteNode(package)) == $(QuoteNode(PKG_THREADS))
                    A  = @zeros(4, 5, 6)
                    @parallel_indices (ix,iy,iz) function test_macros!(A)
                        @test @gridDim() == Dim3(2, 3, 6)
                        @test @blockIdx() == Dim3(ix-2, iy-1, iz)
                        @test @blockDim() == Dim3(1, 1, 1)
                        @test @threadIdx() == Dim3(1, 1, 1)
                        return
                    end
                    @parallel (3:4, 2:4, 1:6) test_macros!(A);
                    @test true # @gridDim test succeeded if this line is reached (the above tests within the kernel are not captured if they succeed, only if they fail...; alternative test implementations would be of course possible, but would be more complex).
                    @test true # @blockIdx ...
                    @test true # @blockDim ...
                    @test true # @threadIdx ...
                end
            end;
            @testset "sync_threads" begin
                @static if $(QuoteNode(package)) == $(QuoteNode(PKG_THREADS))
                    @test @prettystring(ParallelStencil.ParallelKernel.@sync_threads_cpu()) == "begin\nend"
                end;
            end;
            @testset "shared memory (allocation)" begin
                @static if $(QuoteNode(package)) == $(QuoteNode(PKG_THREADS))
                    @test typeof(@sharedMem(Float32,(2,3))) == typeof(ParallelStencil.ParallelKernel.MArray{Tuple{2,3},   Float32, length((2,3)),   prod((2,3))}(undef))
                    @test typeof(@sharedMem(Bool,(2,3,4)))  == typeof(ParallelStencil.ParallelKernel.MArray{Tuple{2,3,4}, Bool,    length((2,3,4)), prod((2,3,4))}(undef))
                end;
            end;
            @testset "@sharedMem (1D)" begin
                @static if $(QuoteNode(package)) == $(QuoteNode(PKG_THREADS))
                    A  = @rand(4)
                    B  = @zeros(4)
                    @parallel_indices (ix) function memcopy!(B, A)
                        tx  = @threadIdx().x
                        A_l = @sharedMem(eltype(A), (@blockDim().x))
                        A_l[tx] = A[ix]
                        @sync_threads()
                        B[ix] = A_l[tx]
                        return
                    end
                    @parallel memcopy!(B, A);
                    @test B == A
                end
            end;
            @testset "@sharedMem (2D)" begin
                @static if $(QuoteNode(package)) == $(QuoteNode(PKG_THREADS))
                    A  = @rand(4,5)
                    B  = @zeros(4,5)
                    @parallel_indices (ix,iy) function memcopy!(B, A)
                        tx  = @threadIdx().x
                        ty  = @threadIdx().y
                        A_l = @sharedMem(eltype(A), (@blockDim().x, @blockDim().y))
                        A_l[tx,ty] = A[ix,iy]
                        @sync_threads()
                        B[ix,iy] = A_l[tx,ty]
                        return
                    end
                    @parallel memcopy!(B, A);
                    @test B == A
                end
            end;
            @testset "@sharedMem (3D)" begin
                @static if $(QuoteNode(package)) == $(QuoteNode(PKG_THREADS))
                    A  = @rand(4,5,6)
                    B  = @zeros(4,5,6)
                    @parallel_indices (ix,iy,iz) function memcopy!(B, A)
                        tx  = @threadIdx().x
                        ty  = @threadIdx().y
                        tz  = @threadIdx().z
                        A_l = @sharedMem(eltype(A), (@blockDim().x, @blockDim().y, @blockDim().z))
                        A_l[tx,ty,tz] = A[ix,iy,iz]
                        @sync_threads()
                        B[ix,iy,iz] = A_l[tx,ty,tz]
                        return
                    end
                    @parallel memcopy!(B, A);
                    @test B == A
                end
            end;
            @reset_parallel_kernel()
        end;
        @testset "2. Exceptions" begin
            @init_parallel_kernel($package, Float64)
            @require @is_initialized
            @testset "no arguments" begin
                @test_throws ArgumentError checknoargs(:(something));                                      # Error: length(args) != 0
            end;
            @testset "arguments @sharedMem" begin
                @test_throws ArgumentError checkargs_sharedMem();                                          # Error: isempty(args)
                @test_throws ArgumentError checkargs_sharedMem(:(something));                              # Error: length(args) != 2
                @test_throws ArgumentError checkargs_sharedMem(:(something), :(something), :(something));  # Error: length(args) != 2
            end;
            @reset_parallel_kernel()
        end;
    end;
)) end == nothing || true;
