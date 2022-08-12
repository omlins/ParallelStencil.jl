using Test
using CellArrays, StaticArrays
import ParallelStencil
using ParallelStencil.ParallelKernel
import ParallelStencil.ParallelKernel: @reset_parallel_kernel, @is_initialized, @get_numbertype, NUMBERTYPE_NONE, SUPPORTED_PACKAGES, PKG_CUDA
import ParallelStencil.ParallelKernel: @require
TEST_PACKAGES = SUPPORTED_PACKAGES
@static if PKG_CUDA in TEST_PACKAGES
    import CUDA
    if !CUDA.functional() TEST_PACKAGES = filter!(x->x≠PKG_CUDA, TEST_PACKAGES) end
end

@static for package in TEST_PACKAGES  eval(:(
    @testset "$(basename(@__FILE__)) (package: $(nameof($package)))" begin
        @testset "1. allocator macros (with default numbertype)" begin
            @require !@is_initialized()
            @init_parallel_kernel($package, Float16)
            @require @is_initialized()
            @testset "mapping to package (no celldims)" begin
                @test @zeros(2,3)                 == parentmodule($package).zeros(Float16,2,3)
                @test @zeros(2,3, eltype=Float32) == parentmodule($package).zeros(Float32,2,3)
                @test @ones(2,3)                  == parentmodule($package).ones(Float16,2,3)
                @test @ones(2,3, eltype=Float32)  == parentmodule($package).ones(Float32,2,3)
                @static if $package == $PKG_CUDA
                    @test typeof(@rand(2,3))                    == typeof(CUDA.CuArray(rand(Float16,2,3)))
                    @test typeof(@rand(2,3, eltype=Float64))    == typeof(CUDA.CuArray(rand(Float64,2,3)))
                    @test typeof(@fill(9, 2,3))                 == typeof(CUDA.CuArray(fill(convert(Float16, 9), 2,3)))
                    @test typeof(@fill(9, 2,3, eltype=Float64)) == typeof(CUDA.CuArray(fill(convert(Float64, 9), 2,3)))
                else
                    @test typeof(@rand(2,3))                    == typeof(parentmodule($package).rand(Float16,2,3))
                    @test typeof(@rand(2,3, eltype=Float64))    == typeof(parentmodule($package).rand(Float64,2,3))
                    @test typeof(@fill(9, 2,3))                 == typeof(fill(convert(Float16, 9), 2,3))
                    @test typeof(@fill(9, 2,3, eltype=Float64)) == typeof(fill(convert(Float64, 9), 2,3))
                end
                @test @falses(2,3) == parentmodule($package).falses(2,3)
                @test @trues(2,3)  == parentmodule($package).trues(2,3)
            end;
            @testset "mapping to package (with celldims)" begin
                T_Float16 = SMatrix{(3,4)..., Float16, prod((3,4))}
                T_Float32 = SMatrix{(3,4)..., Float32, prod((3,4))}
                T_Float64 = SMatrix{(3,4)..., Float64, prod((3,4))}
                T_Bool    = SMatrix{(3,4)..., Bool, prod((3,4))}
                @static if $package == $PKG_CUDA
                    CUDA.allowscalar(true)
                    @test @zeros(2,3, celldims=(3,4))                           == CellArrays.fill!(CuCellArray{T_Float16}(undef,2,3), T_Float16(zeros((3,4))))
                    @test @zeros(2,3, celldims=(3,4), eltype=Float32)           == CellArrays.fill!(CuCellArray{T_Float32}(undef,2,3), T_Float32(zeros((3,4))))
                    @test @ones(2,3, celldims=(3,4))                            == CellArrays.fill!(CuCellArray{T_Float16}(undef,2,3), T_Float16(ones((3,4))))
                    @test @ones(2,3, celldims=(3,4), eltype=Float32)            == CellArrays.fill!(CuCellArray{T_Float32}(undef,2,3), T_Float32(ones((3,4))))
                    @test typeof(@rand(2,3, celldims=(3,4)))                    == typeof(CuCellArray{T_Float16,0}(undef,2,3))
                    @test typeof(@rand(2,3, celldims=(3,4), eltype=Float64))    == typeof(CuCellArray{T_Float64,0}(undef,2,3))
                    @test typeof(@fill(9, 2,3, celldims=(3,4)))                 == typeof(CuCellArray{T_Float16,0}(undef,2,3))
                    @test typeof(@fill(9, 2,3, celldims=(3,4), eltype=Float64)) == typeof(CuCellArray{T_Float64,0}(undef,2,3))
                    @test @falses(2,3, celldims=(3,4))                          == CellArrays.fill!(CuCellArray{T_Bool}(undef,2,3), falses((3,4)))
                    @test @trues(2,3, celldims=(3,4))                           == CellArrays.fill!(CuCellArray{T_Bool}(undef,2,3), trues((3,4)))
                    CUDA.allowscalar(false)
                else
                    @test @zeros(2,3, celldims=(3,4))                           == CellArrays.fill!(CPUCellArray{T_Float16}(undef,2,3), T_Float16(zeros((3,4))))
                    @test @zeros(2,3, celldims=(3,4), eltype=Float32)           == CellArrays.fill!(CPUCellArray{T_Float32}(undef,2,3), T_Float32(zeros((3,4))))
                    @test @ones(2,3, celldims=(3,4))                            == CellArrays.fill!(CPUCellArray{T_Float16}(undef,2,3), T_Float16(ones((3,4))))
                    @test @ones(2,3, celldims=(3,4), eltype=Float32)            == CellArrays.fill!(CPUCellArray{T_Float32}(undef,2,3), T_Float32(ones((3,4))))
                    @test typeof(@rand(2,3, celldims=(3,4)))                    == typeof(CPUCellArray{T_Float16,1}(undef,2,3))
                    @test typeof(@rand(2,3, celldims=(3,4), eltype=Float64))    == typeof(CPUCellArray{T_Float64,1}(undef,2,3))
                    @test typeof(@fill(9, 2,3, celldims=(3,4)))                 == typeof(CPUCellArray{T_Float16,1}(undef,2,3))
                    @test typeof(@fill(9, 2,3, celldims=(3,4), eltype=Float64)) == typeof(CPUCellArray{T_Float64,1}(undef,2,3))
                    @test @falses(2,3, celldims=(3,4))                          == CellArrays.fill!(CPUCellArray{T_Bool}(undef,2,3), falses((3,4)))
                    @test @trues(2,3, celldims=(3,4))                           == CellArrays.fill!(CPUCellArray{T_Bool}(undef,2,3), trues((3,4)))
                end
            end;
            @reset_parallel_kernel()
        end;
        @testset "2. allocator macros (no default numbertype)" begin            # Note: these tests are exact copies of 1. with the tests without eltype kwarg removed though (i.e., every 2nd test removed)
            @require !@is_initialized()
            @init_parallel_kernel(package = $package)
            @require @is_initialized()
            @require @get_numbertype() == NUMBERTYPE_NONE
            @testset "mapping to package (no celldims)" begin
                @test @zeros(2,3, eltype=Float32) == parentmodule($package).zeros(Float32,2,3)
                @test @ones(2,3, eltype=Float32)  == parentmodule($package).ones(Float32,2,3)
                @static if $package == $PKG_CUDA
                    @test typeof(@rand(2,3, eltype=Float64))    == typeof(CUDA.CuArray(rand(Float64,2,3)))
                    @test typeof(@fill(9, 2,3, eltype=Float64)) == typeof(CUDA.CuArray(fill(convert(Float64, 9), 2,3)))
                else
                    @test typeof(@rand(2,3, eltype=Float64))    == typeof(parentmodule($package).rand(Float64,2,3))
                    @test typeof(@fill(9, 2,3, eltype=Float64)) == typeof(fill(convert(Float64, 9), 2,3))
                end
                @test @falses(2,3) == parentmodule($package).falses(2,3)
                @test @trues(2,3)  == parentmodule($package).trues(2,3)
            end;
            @testset "mapping to package (with celldims)" begin
                T_Float16 = SMatrix{(3,4)..., Float16, prod((3,4))}
                T_Float32 = SMatrix{(3,4)..., Float32, prod((3,4))}
                T_Float64 = SMatrix{(3,4)..., Float64, prod((3,4))}
                T_Bool    = SMatrix{(3,4)..., Bool, prod((3,4))}
                @static if $package == $PKG_CUDA
                    CUDA.allowscalar(true)
                    @test @zeros(2,3, celldims=(3,4), eltype=Float32)           == CellArrays.fill!(CuCellArray{T_Float32}(undef,2,3), T_Float32(zeros((3,4))))
                    @test @ones(2,3, celldims=(3,4), eltype=Float32)            == CellArrays.fill!(CuCellArray{T_Float32}(undef,2,3), T_Float32(ones((3,4))))
                    @test typeof(@rand(2,3, celldims=(3,4), eltype=Float64))    == typeof(CuCellArray{T_Float64,0}(undef,2,3))
                    @test typeof(@fill(9, 2,3, celldims=(3,4), eltype=Float64)) == typeof(CuCellArray{T_Float64,0}(undef,2,3))
                    @test @falses(2,3, celldims=(3,4))                          == CellArrays.fill!(CuCellArray{T_Bool}(undef,2,3), falses((3,4)))
                    @test @trues(2,3, celldims=(3,4))                           == CellArrays.fill!(CuCellArray{T_Bool}(undef,2,3), trues((3,4)))
                    CUDA.allowscalar(false)
                else
                    @test @zeros(2,3, celldims=(3,4), eltype=Float32)           == CellArrays.fill!(CPUCellArray{T_Float32}(undef,2,3), T_Float32(zeros((3,4))))
                    @test @ones(2,3, celldims=(3,4), eltype=Float32)            == CellArrays.fill!(CPUCellArray{T_Float32}(undef,2,3), T_Float32(ones((3,4))))
                    @test typeof(@rand(2,3, celldims=(3,4), eltype=Float64))    == typeof(CPUCellArray{T_Float64,1}(undef,2,3))
                    @test typeof(@fill(9, 2,3, celldims=(3,4), eltype=Float64)) == typeof(CPUCellArray{T_Float64,1}(undef,2,3))
                    @test @falses(2,3, celldims=(3,4))                          == CellArrays.fill!(CPUCellArray{T_Bool}(undef,2,3), falses((3,4)))
                    @test @trues(2,3, celldims=(3,4))                           == CellArrays.fill!(CPUCellArray{T_Bool}(undef,2,3), trues((3,4)))
                end
            end;
            @reset_parallel_kernel()
        end;
    end;
)) end == nothing || true;
