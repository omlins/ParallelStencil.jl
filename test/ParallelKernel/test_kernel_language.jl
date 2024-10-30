using Test
import ParallelStencil
using ParallelStencil.ParallelKernel
import ParallelStencil.ParallelKernel: @reset_parallel_kernel, @is_initialized, SUPPORTED_PACKAGES, PKG_CUDA, PKG_AMDGPU, PKG_METAL, PKG_THREADS, PKG_POLYESTER
import ParallelStencil.ParallelKernel: @require, @prettystring, @iscpu
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
@static if PKG_METAL in TEST_PACKAGES
    import Metal
    if !Metal.functional() TEST_PACKAGES = filter!(x->x≠PKG_METAL, TEST_PACKAGES) end
end
@static if PKG_POLYESTER in TEST_PACKAGES
    import Polyester
end
Base.retry_load_extensions() # Potentially needed to load the extensions after the packages have been filtered.

const TEST_PRECISIONS = [Float32, Float64]
@static for package in TEST_PACKAGES
for precision in TEST_PRECISIONS
(package == PKG_METAL && precision == Float64) ? continue : nothing # Metal does not support Float64
    
eval(:(
    @testset "$(basename(@__FILE__)) (package: $(nameof($package))) (precision: $(nameof($precision)))" begin
        @testset "1. kernel language macros" begin
            @require !@is_initialized()
            @init_parallel_kernel($package, $precision)
            @require @is_initialized()
            @testset "mapping to package" begin
                if $package == $PKG_CUDA
                    @test @prettystring(1, @gridDim()) == "CUDA.gridDim()"
                    @test @prettystring(1, @blockIdx()) == "CUDA.blockIdx()"
                    @test @prettystring(1, @blockDim()) == "CUDA.blockDim()"
                    @test @prettystring(1, @threadIdx()) == "CUDA.threadIdx()"
                    @test @prettystring(1, @sync_threads()) == "CUDA.sync_threads()"
                    @test @prettystring(1, @sharedMem($precision, (2,3))) == "CUDA.@cuDynamicSharedMem $(nameof($precision)) (2, 3)"
                    # @test @prettystring(1, @pk_show()) == "CUDA.@cushow"
                    # @test @prettystring(1, @pk_println()) == "CUDA.@cuprintln"
                elseif $package == $AMDGPU
                    @test @prettystring(1, @gridDim()) == "AMDGPU.gridGroupDim()"
                    @test @prettystring(1, @blockIdx()) == "AMDGPU.workgroupIdx()"
                    @test @prettystring(1, @blockDim()) == "AMDGPU.workgroupDim()"
                    @test @prettystring(1, @threadIdx()) == "AMDGPU.workitemIdx()"
                    @test @prettystring(1, @sync_threads()) == "AMDGPU.sync_workgroup()"
                    # @test @prettystring(1, @sharedMem($precision, (2,3))) == ""    #TODO: not yet supported for AMDGPU
                    # @test @prettystring(1, @pk_show()) == "CUDA.@cushow"        #TODO: not yet supported for AMDGPU
                    # @test @prettystring(1, @pk_println()) == "AMDGPU.@rocprintln"
                elseif $package == $PKG_METAL
                    @test @prettystring(1, @gridDim()) == "Metal.threadgroups_per_grid_3d()"
                    @test @prettystring(1, @blockIdx()) == "Metal.threadgroup_position_in_grid_3d()"
                    @test @prettystring(1, @blockDim()) == "Metal.threads_per_threadgroup_3d()"
                    @test @prettystring(1, @threadIdx()) == "Metal.thread_position_in_threadgroup_3d()"
                    @test @prettystring(1, @sync_threads()) == "Metal.threadgroup_barrier(; flag = Metal.MemoryFlagThreadGroup)"
                    @test @prettystring(1, @sharedMem($precision, (2,3))) == "ParallelStencil.ParallelKernel.@sharedMem_metal $(nameof($precision)) (2, 3)"
                    # @test @prettystring(1, @pk_show()) == "Metal.@mtlshow"
                    # @test @prettystring(1, @pk_println()) == "Metal.@mtlprintln"
                elseif @iscpu($package)
                    @test @prettystring(1, @gridDim()) == "ParallelStencil.ParallelKernel.@gridDim_cpu"
                    @test @prettystring(1, @blockIdx()) == "ParallelStencil.ParallelKernel.@blockIdx_cpu"
                    @test @prettystring(1, @blockDim()) == "ParallelStencil.ParallelKernel.@blockDim_cpu"
                    @test @prettystring(1, @threadIdx()) == "ParallelStencil.ParallelKernel.@threadIdx_cpu"
                    @test @prettystring(1, @sync_threads()) == "ParallelStencil.ParallelKernel.@sync_threads_cpu"
                    @test @prettystring(1, @sharedMem($precision, (2,3))) == "ParallelStencil.ParallelKernel.@sharedMem_cpu $(nameof($precision)) (2, 3)"
                    # @test @prettystring(1, @pk_show()) == "Base.@show"
                    # @test @prettystring(1, @pk_println()) == "Base.println()"
                end;
            end;
            @testset "mapping to package (internal macros)" begin
                if $package == $PKG_THREADS
                    @test @prettystring(1, ParallelStencil.ParallelKernel.@threads()) == "Base.Threads.@threads"
                elseif $package == $PKG_POLYESTER
                    @test @prettystring(1, ParallelStencil.ParallelKernel.@threads()) == "Polyester.@batch"
                end;
            end;
            @testset "@gridDim, @blockIdx, @blockDim, @threadIdx (1D)" begin
                @static if $package == $PKG_THREADS
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
                @static if $package == $PKG_THREADS
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
                @static if $package == $PKG_THREADS
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
                @static if @iscpu($package)
                    @test @prettystring(ParallelStencil.ParallelKernel.@sync_threads_cpu()) == "begin\nend"
                end;
            end;
            @testset "shared memory (allocation)" begin
                @static if @iscpu($package)
                    @test typeof(@sharedMem($precision,(2,3))) == typeof(ParallelStencil.ParallelKernel.MArray{Tuple{2,3},   $precision, length((2,3)),   prod((2,3))}(undef))
                    @test typeof(@sharedMem(Bool,(2,3,4)))  == typeof(ParallelStencil.ParallelKernel.MArray{Tuple{2,3,4}, Bool,    length((2,3,4)), prod((2,3,4))}(undef))
                end;
            end;
            @testset "@sharedMem (1D)" begin
                @static if @iscpu($package)
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
                @static if @iscpu($package)
                    A  = @rand(4,5)
                    B  = @zeros(4,5)
                    @parallel_indices (ix,iy) function memcopy!(B, A)
                        tx  = @threadIdx().x
                        ty  = @threadIdx().y
                        A_l = @sharedMem(eltype(A), (@blockDim().x, @blockDim().y), 0*sizeof(eltype(A)))
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
                @static if @iscpu($package)
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
            @testset "@∀" begin
                expansion = @prettystring(1, @∀ i ∈ (x,z) @all(C.i) = @all(A.i) + @all(B.i))
                @test occursin("@all(C.x) = @all(A.x) + @all(B.x)", expansion)
                @test occursin("@all(C.z) = @all(A.z) + @all(B.z)", expansion)
                expansion = @prettystring(1, @∀ i ∈ (y,z) C.i[ix,iy,iz] = A.i[ix,iy,iz] + B.i[ix,iy,iz])
                @test occursin("C.y[ix, iy, iz] = A.y[ix, iy, iz] + B.y[ix, iy, iz]", expansion)
                @test occursin("C.z[ix, iy, iz] = A.z[ix, iy, iz] + B.z[ix, iy, iz]", expansion)
                expansion = @prettystring(1, @∀ (ij,i,j) ∈ ((xy,x,y), (xz,x,z), (yz,y,z)) @all(C.ij) = @all(A.i) + @all(B.j))
                @test occursin("@all(C.xy) = @all(A.x) + @all(B.y)", expansion)
                @test occursin("@all(C.xz) = @all(A.x) + @all(B.z)", expansion)
                @test occursin("@all(C.yz) = @all(A.y) + @all(B.z)", expansion)
                expansion = @prettystring(1, @∀ i ∈ 1:N-1 @all(C[i]) = @all(A[i]) + @all(B[i]))
                @test occursin("ntuple(Val(N - 1)) do i", expansion)
                @test occursin("@all(C[i]) = @all(A[i]) + @all(B[i])", expansion)
                expansion = @prettystring(1, @∀ i ∈ 2:N-1 C[i][ix,iy,iz] = A[i][ix,iy,iz] + B[i][ix,iy,iz])
                @test occursin("ntuple(Val(((N - 1) - 2) + 1)) do i", expansion)
                @test occursin("(C[(i + 2) - 1])[ix, iy, iz] = (A[(i + 2) - 1])[ix, iy, iz] + (B[(i + 2) - 1])[ix, iy, iz]", expansion)
            end;
            @reset_parallel_kernel()
        end;
        @testset "2. Exceptions" begin
            @init_parallel_kernel($package, $precision)
            @require @is_initialized
            @testset "no arguments" begin
                @test_throws ArgumentError checknoargs(:(something));                                                   # Error: length(args) != 0
            end;
            @testset "arguments @sharedMem" begin
                @test_throws ArgumentError checkargs_sharedMem();                                                        # Error: isempty(args)
                @test_throws ArgumentError checkargs_sharedMem(:(something));                                            # Error: length(args) != 2
                @test_throws ArgumentError checkargs_sharedMem(:(something), :(something), :(something), :(something));  # Error: length(args) != 2
            end;
            @reset_parallel_kernel()
        end;
    end;
))

end end == nothing || true;
