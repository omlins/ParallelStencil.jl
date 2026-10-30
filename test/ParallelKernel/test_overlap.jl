using Test
import ParallelStencil
using ParallelStencil.ParallelKernel
import ParallelStencil.ParallelKernel: @reset_parallel_kernel, @is_initialized, SUPPORTED_PACKAGES, PKG_CUDA, PKG_AMDGPU, PKG_METAL, PKG_POLYESTER
import ParallelStencil.ParallelKernel: @require, @gorgeousexpand, @isgpu
import ParallelStencil.ParallelKernel: checkargs_overlap, @overlap, overlap_gpu
using ParallelStencil.ParallelKernel.Exceptions
include(joinpath(@__DIR__, "../packages.jl")) # Selection of the packages to test (TEST_PACKAGES), imports of the backend packages.

@testset "$(basename(@__FILE__))" verbose=true begin # NOTE: the package testsets are nested in this testset (and the package loop is run at run time) in order to test all packages even if the tests of one package fail.
for package in TEST_PACKAGES
    FloatDefault = (package == PKG_METAL) ? Float32 : Float64 # Metal does not support Float64
    
eval(:(
    @testset "$(basename(@__FILE__)) (package: $(nameof($package)))" begin
        @testset "1. overlap macro" begin
            @require !@is_initialized()
            @init_parallel_kernel($package, $FloatDefault)
            @require @is_initialized()
            @testset "@overlap block (macro expansion)" begin
                @static if @isgpu($package)
                    block = string(@gorgeousexpand(1, @overlap begin
                        @parallel (1:sim.Nx, 2:sim.Ny) kernel1!(sim.array1)
                        @parallel (1:sim.Nx, 2:sim.Ny) kernel2!(sim.array2)
                        @parallel (1:sim.Nx, 2:sim.Ny) kernel3!(sim.array3)
                    end))
                    @test occursin("@parallel_async (1:sim.Nx, 2:sim.Ny) stream = ParallelStencil.ParallelKernel.@get_stream(1) kernel1!(sim.array1)", block)
                    @test occursin("@parallel_async (1:sim.Nx, 2:sim.Ny) stream = ParallelStencil.ParallelKernel.@get_stream(2) kernel2!(sim.array2)", block)
                    @test occursin("@parallel_async (1:sim.Nx, 2:sim.Ny) stream = ParallelStencil.ParallelKernel.@get_stream(3) kernel3!(sim.array3)", block)
                    @test occursin("ParallelStencil.ParallelKernel.@synchronize ParallelStencil.ParallelKernel.@get_stream(1)", block)
                    @test occursin("ParallelStencil.ParallelKernel.@synchronize ParallelStencil.ParallelKernel.@get_stream(2)", block)
                    @test occursin("ParallelStencil.ParallelKernel.@synchronize ParallelStencil.ParallelKernel.@get_stream(3)", block)
                    @test occursin("ParallelStencil.ParallelKernel.suspend_implicit_sync(", block)
                    @test occursin("ParallelStencil.ParallelKernel.resume_implicit_sync(", block)
                end;
            end;
            @testset "@overlap execution" begin
                @parallel_indices (ix,iy,iz) function fill_value!(A, value)
                    A[ix,iy,iz] = value
                    return
                end
                A = @zeros(4, 3, 2, eltype=Float32)
                B = @zeros(4, 3, 2, eltype=Float32)
                one = Float32(1)
                two = Float32(2)
                @overlap begin
                    @parallel fill_value!(A, one)
                    @parallel fill_value!(B, two)
                end
                @test all(Array(A) .== one)
                @test all(Array(B) .== two)
            end;
            @testset "@overlap: implicit synchronization of the arrays suspended" begin # NOTE: CUDA.jl (from v6.3) and AMDGPU.jl order every use of an array on a stream after its last use on another stream; unless this is suspended, calls reading the same array run one after the other.
                @static if $(package in (PKG_CUDA, PKG_AMDGPU))
                    has_optout = $(package == PKG_CUDA) ? isdefined(CUDA, :enable_synchronization!) : isdefined(AMDGPU, :enable_synchronization!)
                    @parallel_indices (ix,iy,iz) function slow_copy!(A, B, n)
                        x = B[ix,iy,iz]
                        for _ in 1:n
                            x = x*0.999f0 + 0.001f0
                        end
                        A[ix,iy,iz] = x
                        return
                    end
                    A1 = @zeros(4, 4, 4, eltype=Float32) # NOTE: small arrays, so that two kernels fit on the GPU at the same time.
                    A2 = @zeros(4, 4, 4, eltype=Float32)
                    B  = @ones(4, 4, 4, eltype=Float32)
                    n  = 1
                    @overlap begin # NOTE: warm-up (compilation).
                        @parallel slow_copy!(A1, B, n)
                        @parallel slow_copy!(A2, B, n)
                    end
                    while (@elapsed @parallel slow_copy!(A1, B, n)) < 0.2 n *= 2 end # NOTE: kernels of at least about 0.2 s.
                    t1 = @elapsed @parallel slow_copy!(A1, B, n)
                    t2 = @elapsed @overlap begin
                        @parallel slow_copy!(A1, B, n)
                        @parallel slow_copy!(A2, B, n)
                    end
                    @static if $(package == PKG_CUDA)
                        @test t2 < 1.5*t1 # NOTE: the two calls ran at the same time, although both read B.
                    else
                        if has_optout @test t2 < 1.5*t1 else @test_broken t2 < 1.5*t1 end # NOTE: AMDGPU.jl versions without enable_synchronization! cannot overlap them.
                    end
                end
            end;
            @reset_parallel_kernel()
        end;
        @testset "2. Exceptions" begin
            @require !@is_initialized()
            @init_parallel_kernel($package, $FloatDefault)
            @require @is_initialized()
            @test_throws ArgumentError checkargs_overlap()
            @test_throws ArgumentError checkargs_overlap(:notablock)
            @test_throws ArgumentError overlap_gpu(:(begin end))
            @test_throws ArgumentError overlap_gpu(:(begin not_parallel_call(); end))
            @test_throws ArgumentError overlap_gpu(:(begin a = 99; end))
            @test_throws ArgumentError overlap_gpu(:(begin @parallel stream=mystream f(); end))
            @reset_parallel_kernel()
        end;
    end;
))
end
end;
