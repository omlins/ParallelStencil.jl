using Test
using CellArrays, StaticArrays
import ParallelStencil
using ParallelStencil.ParallelKernel
import ParallelStencil.ParallelKernel: @reset_parallel_kernel, @is_initialized, @get_numbertype, NUMBERTYPE_NONE, SUPPORTED_PACKAGES, PKG_CUDA, PKG_AMDGPU
import ParallelStencil.ParallelKernel: @require, @prettystring, @gorgeousstring
import ParallelStencil.ParallelKernel: checkargs_CellType, _CellType
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
        @testset "1. @CellType macro" begin
            @require !@is_initialized()
            @init_parallel_kernel($package, Float16)
            @require @is_initialized()
            @testset "fieldnames" begin
                call = @prettystring(1, @CellType SymmetricTensor2D fieldnames=(xx, zz, xz))
                @test occursin("struct SymmetricTensor2D <: ParallelStencil.ParallelKernel.FieldArray{Tuple{3}, Float16, length([3])}", call)
                @test occursin("xx::Float16", call)
                @test occursin("zz::Float16", call)
                @test occursin("xz::Float16", call)
                call = @prettystring(1, @CellType SymmetricTensor3D fieldnames=(xx, yy, zz, yz, xz, xy))
                @test occursin("struct SymmetricTensor3D <: ParallelStencil.ParallelKernel.FieldArray{Tuple{6}, Float16, length([6])}", call)
                @test occursin("xx::Float16", call)
                @test occursin("yy::Float16", call)
                @test occursin("zz::Float16", call)
                @test occursin("yz::Float16", call)
                @test occursin("xz::Float16", call)
                @test occursin("xy::Float16", call)
            end;
            @testset "dims" begin
                call = @prettystring(1, @CellType Tensor2D fieldnames=(xxxx, yxxx, xyxx, yyxx, xxyx, yxyx, xyyx, yyyx, xxxy, yxxy, xyxy, yyxy, xxyy, yxyy, xyyy, yyyy) dims=(2,2,2,2))
                @test occursin("struct Tensor2D <: ParallelStencil.ParallelKernel.FieldArray{Tuple{2, 2, 2, 2}, Float16, length(Any[2, 2, 2, 2])}", call)
                @test occursin("xxxx::Float16", call)
                @test occursin("yxxx::Float16", call)
                @test occursin("xyxx::Float16", call)
                @test occursin("yyxx::Float16", call)
                @test occursin("xxyx::Float16", call)
                @test occursin("yxyx::Float16", call)
                @test occursin("xyyx::Float16", call)
                @test occursin("yyyx::Float16", call)
                @test occursin("xxxy::Float16", call)
                @test occursin("yxxy::Float16", call)
                @test occursin("xyxy::Float16", call)
                @test occursin("yyxy::Float16", call)
                @test occursin("xxyy::Float16", call)
                @test occursin("yxyy::Float16", call)
                @test occursin("xyyy::Float16", call)
                @test occursin("yyyy::Float16", call)
            end;
            @testset "parametric" begin
                call = @prettystring(1, @CellType SymmetricTensor2D fieldnames=(xx, zz, xz) parametric=true)
                @test occursin("struct SymmetricTensor2D{T} <: ParallelStencil.ParallelKernel.FieldArray{Tuple{3}, T, length([3])}", call)
                @test occursin("xx::T", call)
                @test occursin("zz::T", call)
                @test occursin("xz::T", call)
            end;
            @testset "eltype" begin
                call = @prettystring(1, @CellType SymmetricTensor2D fieldnames=(xx, zz, xz) eltype=Float32)
                @test occursin("struct SymmetricTensor2D <: ParallelStencil.ParallelKernel.FieldArray{Tuple{3}, Float32, length([3])}", call)
                @test occursin("xx::Float32", call)
                @test occursin("zz::Float32", call)
                @test occursin("xz::Float32", call)
            end;
            @reset_parallel_kernel()
        end;
        @testset "2. allocator macros (with default numbertype)" begin
            @require !@is_initialized()
            @init_parallel_kernel($package, Float16)
            @require @is_initialized()
            @testset "datatype definitions" begin
                @CellType SymmetricTensor2D fieldnames=(xx, zz, xz)
                @CellType SymmetricTensor3D fieldnames=(xx, yy, zz, yz, xz, xy)
                @CellType Tensor2D fieldnames=(xxxx, yxxx, xyxx, yyxx, xxyx, yxyx, xyyx, yyyx, xxxy, yxxy, xyxy, yyxy, xxyy, yxyy, xyyy, yyyy) dims=(2,2,2,2)
                @CellType SymmetricTensor2D_T fieldnames=(xx, zz, xz) parametric=true
                @CellType SymmetricTensor2D_Float32 fieldnames=(xx, zz, xz) eltype=Float32
                @CellType SymmetricTensor2D_Bool fieldnames=(xx, zz, xz) eltype=Bool
                @test SymmetricTensor2D <: FieldArray
                @test SymmetricTensor3D <: FieldArray
                @test Tensor2D <: FieldArray
                @test SymmetricTensor2D_T <: FieldArray
                @test SymmetricTensor2D_Float32 <: FieldArray
                @test SymmetricTensor2D_Bool <: FieldArray
            end;
            @testset "mapping to package (no celldims/celltype)" begin
                @test @zeros(2,3)                 == parentmodule($package).zeros(Float16,2,3)
                @test @zeros(2,3, eltype=Float32) == parentmodule($package).zeros(Float32,2,3)
                @test @ones(2,3)                  == parentmodule($package).ones(Float16,2,3)
                @test @ones(2,3, eltype=Float32)  == parentmodule($package).ones(Float32,2,3)
                @static if $package == $PKG_CUDA
                    @test typeof(@rand(2,3))                    == typeof(CUDA.CuArray(rand(Float16,2,3)))
                    @test typeof(@rand(2,3, eltype=Float64))    == typeof(CUDA.CuArray(rand(Float64,2,3)))
                    @test typeof(@fill(9, 2,3))                 == typeof(CUDA.CuArray(fill(convert(Float16, 9), 2,3)))
                    @test typeof(@fill(9, 2,3, eltype=Float64)) == typeof(CUDA.CuArray(fill(convert(Float64, 9), 2,3)))
                elseif $package == $PKG_AMDGPU
                    @test typeof(@rand(2,3))                    == typeof(AMDGPU.ROCArray(rand(Float16,2,3)))
                    @test typeof(@rand(2,3, eltype=Float64))    == typeof(AMDGPU.ROCArray(rand(Float64,2,3)))
                    @test typeof(@fill(9, 2,3))                 == typeof(AMDGPU.ROCArray(fill(convert(Float16, 9), 2,3)))
                    @test typeof(@fill(9, 2,3, eltype=Float64)) == typeof(AMDGPU.ROCArray(fill(convert(Float64, 9), 2,3)))
                else
                    @test typeof(@rand(2,3))                    == typeof(parentmodule($package).rand(Float16,2,3))
                    @test typeof(@rand(2,3, eltype=Float64))    == typeof(parentmodule($package).rand(Float64,2,3))
                    @test typeof(@fill(9, 2,3))                 == typeof(fill(convert(Float16, 9), 2,3))
                    @test typeof(@fill(9, 2,3, eltype=Float64)) == typeof(fill(convert(Float64, 9), 2,3))
                end
                @test Array(@falses(2,3)) == Array(parentmodule($package).falses(2,3))
                @test Array(@trues(2,3))  == Array(parentmodule($package).trues(2,3))
            end;
            @testset "mapping to package (with celldims)" begin
                T_Float16 = SMatrix{(3,4)..., Float16, prod((3,4))}
                T_Float32 = SMatrix{(3,4)..., Float32, prod((3,4))}
                T_Float64 = SMatrix{(3,4)..., Float64, prod((3,4))}
                T_Bool    = SMatrix{(3,4)..., Bool,    prod((3,4))}
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
                elseif $package == $PKG_AMDGPU
                    AMDGPU.allowscalar(true) #TODO: check how to do (everywhere) (for GPU, CellArray B is the same - could potentially be merged if not using type alias...)
                    @test @zeros(2,3, celldims=(3,4))                           == CellArrays.fill!(ROCCellArray{T_Float16}(undef,2,3), T_Float16(zeros((3,4))))
                    @test @zeros(2,3, celldims=(3,4), eltype=Float32)           == CellArrays.fill!(ROCCellArray{T_Float32}(undef,2,3), T_Float32(zeros((3,4))))
                    @test @ones(2,3, celldims=(3,4))                            == CellArrays.fill!(ROCCellArray{T_Float16}(undef,2,3), T_Float16(ones((3,4))))
                    @test @ones(2,3, celldims=(3,4), eltype=Float32)            == CellArrays.fill!(ROCCellArray{T_Float32}(undef,2,3), T_Float32(ones((3,4))))
                    @test typeof(@rand(2,3, celldims=(3,4)))                    == typeof(ROCCellArray{T_Float16,0}(undef,2,3))
                    @test typeof(@rand(2,3, celldims=(3,4), eltype=Float64))    == typeof(ROCCellArray{T_Float64,0}(undef,2,3))
                    @test typeof(@fill(9, 2,3, celldims=(3,4)))                 == typeof(ROCCellArray{T_Float16,0}(undef,2,3))
                    @test typeof(@fill(9, 2,3, celldims=(3,4), eltype=Float64)) == typeof(ROCCellArray{T_Float64,0}(undef,2,3))
                    @test @falses(2,3, celldims=(3,4))                          == CellArrays.fill!(ROCCellArray{T_Bool}(undef,2,3), falses((3,4)))
                    @test @trues(2,3, celldims=(3,4))                           == CellArrays.fill!(ROCCellArray{T_Bool}(undef,2,3), trues((3,4)))
                    AMDGPU.allowscalar(false) #TODO: check how to do
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
            @testset "mapping to package (with celltype)" begin
                @static if $package == $PKG_CUDA
                    CUDA.allowscalar(true)
                    @test @zeros(2,3, celltype=SymmetricTensor2D)               == CellArrays.fill!(CuCellArray{SymmetricTensor2D}(undef,2,3), SymmetricTensor2D(zeros(3)))
                    @test @zeros(2,3, celltype=SymmetricTensor3D)               == CellArrays.fill!(CuCellArray{SymmetricTensor3D}(undef,2,3), SymmetricTensor3D(zeros(6)))
                    @test @zeros(2,3, celltype=Tensor2D)                        == CellArrays.fill!(CuCellArray{Tensor2D}(undef,2,3), Tensor2D(zeros((2,2,2,2))))
                    @test @zeros(2,3, celltype=SymmetricTensor2D_T{Float64})    == CellArrays.fill!(CuCellArray{SymmetricTensor2D_T{Float64}}(undef,2,3), SymmetricTensor2D_T{Float64}(zeros(3)))
                    @test @zeros(2,3, celltype=SymmetricTensor2D_Float32)       == CellArrays.fill!(CuCellArray{SymmetricTensor2D_Float32}(undef,2,3), SymmetricTensor2D_Float32(zeros(3)))
                    @test @ones(2,3, celltype=SymmetricTensor2D)                == CellArrays.fill!(CuCellArray{SymmetricTensor2D}(undef,2,3), SymmetricTensor2D(ones(3)))
                    @test typeof(@rand(2,3, celltype=SymmetricTensor2D))        == typeof(CuCellArray{SymmetricTensor2D,0}(undef,2,3))
                    @test typeof(@fill(9, 2,3, celltype=SymmetricTensor2D))     == typeof(CuCellArray{SymmetricTensor2D,0}(undef,2,3))
                    CUDA.allowscalar(false)
                elseif $package == $PKG_AMDGPU
                    AMDGPU.allowscalar(true)
                    @test @zeros(2,3, celltype=SymmetricTensor2D)               == CellArrays.fill!(ROCCellArray{SymmetricTensor2D}(undef,2,3), SymmetricTensor2D(zeros(3)))
                    @test @zeros(2,3, celltype=SymmetricTensor3D)               == CellArrays.fill!(ROCCellArray{SymmetricTensor3D}(undef,2,3), SymmetricTensor3D(zeros(6)))
                    @test @zeros(2,3, celltype=Tensor2D)                        == CellArrays.fill!(ROCCellArray{Tensor2D}(undef,2,3), Tensor2D(zeros((2,2,2,2))))
                    @test @zeros(2,3, celltype=SymmetricTensor2D_T{Float64})    == CellArrays.fill!(ROCCellArray{SymmetricTensor2D_T{Float64}}(undef,2,3), SymmetricTensor2D_T{Float64}(zeros(3)))
                    @test @zeros(2,3, celltype=SymmetricTensor2D_Float32)       == CellArrays.fill!(ROCCellArray{SymmetricTensor2D_Float32}(undef,2,3), SymmetricTensor2D_Float32(zeros(3)))
                    @test @ones(2,3, celltype=SymmetricTensor2D)                == CellArrays.fill!(ROCCellArray{SymmetricTensor2D}(undef,2,3), SymmetricTensor2D(ones(3)))
                    @test typeof(@rand(2,3, celltype=SymmetricTensor2D))        == typeof(ROCCellArray{SymmetricTensor2D,0}(undef,2,3))
                    @test typeof(@fill(9, 2,3, celltype=SymmetricTensor2D))     == typeof(ROCCellArray{SymmetricTensor2D,0}(undef,2,3))
                    AMDGPU.allowscalar(false)
                else
                    @test @zeros(2,3, celltype=SymmetricTensor2D)               == CellArrays.fill!(CPUCellArray{SymmetricTensor2D}(undef,2,3), SymmetricTensor2D(zeros(3)))
                    @test @zeros(2,3, celltype=SymmetricTensor3D)               == CellArrays.fill!(CPUCellArray{SymmetricTensor3D}(undef,2,3), SymmetricTensor3D(zeros(6)))
                    @test @zeros(2,3, celltype=Tensor2D)                        == CellArrays.fill!(CPUCellArray{Tensor2D}(undef,2,3), Tensor2D(zeros((2,2,2,2))))
                    @test @zeros(2,3, celltype=SymmetricTensor2D_T{Float64})    == CellArrays.fill!(CPUCellArray{SymmetricTensor2D_T{Float64}}(undef,2,3), SymmetricTensor2D_T{Float64}(zeros(3)))
                    @test @zeros(2,3, celltype=SymmetricTensor2D_Float32)       == CellArrays.fill!(CPUCellArray{SymmetricTensor2D_Float32}(undef,2,3), SymmetricTensor2D_Float32(zeros(3)))
                    @test @ones(2,3, celltype=SymmetricTensor2D)                == CellArrays.fill!(CPUCellArray{SymmetricTensor2D}(undef,2,3), SymmetricTensor2D(ones(3)))
                    @test typeof(@rand(2,3, celltype=SymmetricTensor2D))        == typeof(CPUCellArray{SymmetricTensor2D,1}(undef,2,3))
                    @test typeof(@fill(9, 2,3, celltype=SymmetricTensor2D))     == typeof(CPUCellArray{SymmetricTensor2D,1}(undef,2,3))
                end
            end;
            @reset_parallel_kernel()
        end;
        @testset "3. allocator macros (no default numbertype)" begin            # Note: these tests are exact copies of 1. with the tests without eltype kwarg removed though (i.e., every 2nd test removed)
            @require !@is_initialized()
            @init_parallel_kernel(package = $package)
            @require @is_initialized()
            @require @get_numbertype() == NUMBERTYPE_NONE
            @testset "datatype definitions" begin
                @CellType SymmetricTensor2D fieldnames=(xx, zz, xz) eltype=Float16
                @CellType SymmetricTensor3D fieldnames=(xx, yy, zz, yz, xz, xy) eltype=Float16
                @CellType Tensor2D fieldnames=(xxxx, yxxx, xyxx, yyxx, xxyx, yxyx, xyyx, yyyx, xxxy, yxxy, xyxy, yyxy, xxyy, yxyy, xyyy, yyyy) dims=(2,2,2,2) eltype=Float16
                @CellType SymmetricTensor2D_T fieldnames=(xx, zz, xz) parametric=true
                @CellType SymmetricTensor2D_Float32 fieldnames=(xx, zz, xz) eltype=Float32
                @CellType SymmetricTensor2D_Bool fieldnames=(xx, zz, xz) eltype=Bool
                @test SymmetricTensor2D <: FieldArray
                @test SymmetricTensor3D <: FieldArray
                @test Tensor2D <: FieldArray
                @test SymmetricTensor2D_T <: FieldArray
                @test SymmetricTensor2D_Float32 <: FieldArray
                @test SymmetricTensor2D_Bool <: FieldArray
            end;
            @testset "mapping to package (no celldims/celltype)" begin
                @test @zeros(2,3, eltype=Float32) == parentmodule($package).zeros(Float32,2,3)
                @test @ones(2,3, eltype=Float32)  == parentmodule($package).ones(Float32,2,3)
                @static if $package == $PKG_CUDA
                    @test typeof(@rand(2,3, eltype=Float64))    == typeof(CUDA.CuArray(rand(Float64,2,3)))
                    @test typeof(@fill(9, 2,3, eltype=Float64)) == typeof(CUDA.CuArray(fill(convert(Float64, 9), 2,3)))
                elseif $package == $PKG_AMDGPU
                    @test typeof(@rand(2,3, eltype=Float64))    == typeof(AMDGPU.ROCArray(rand(Float64,2,3)))
                    @test typeof(@fill(9, 2,3, eltype=Float64)) == typeof(AMDGPU.ROCArray(fill(convert(Float64, 9), 2,3)))
                else
                    @test typeof(@rand(2,3, eltype=Float64))    == typeof(parentmodule($package).rand(Float64,2,3))
                    @test typeof(@fill(9, 2,3, eltype=Float64)) == typeof(fill(convert(Float64, 9), 2,3))
                end
                @test Array(@falses(2,3)) == Array(parentmodule($package).falses(2,3))
                @test Array(@trues(2,3))  == Array(parentmodule($package).trues(2,3))
            end;
            @testset "mapping to package (with celldims)" begin
                T_Float16 = SMatrix{(3,4)..., Float16, prod((3,4))}
                T_Float32 = SMatrix{(3,4)..., Float32, prod((3,4))}
                T_Float64 = SMatrix{(3,4)..., Float64, prod((3,4))}
                T_Bool    = SMatrix{(3,4)..., Bool,    prod((3,4))}
                @static if $package == $PKG_CUDA
                    CUDA.allowscalar(true)
                    @test @zeros(2,3, celldims=(3,4), eltype=Float32)           == CellArrays.fill!(CuCellArray{T_Float32}(undef,2,3), T_Float32(zeros((3,4))))
                    @test @ones(2,3, celldims=(3,4), eltype=Float32)            == CellArrays.fill!(CuCellArray{T_Float32}(undef,2,3), T_Float32(ones((3,4))))
                    @test typeof(@rand(2,3, celldims=(3,4), eltype=Float64))    == typeof(CuCellArray{T_Float64,0}(undef,2,3))
                    @test typeof(@fill(9, 2,3, celldims=(3,4), eltype=Float64)) == typeof(CuCellArray{T_Float64,0}(undef,2,3))
                    @test @falses(2,3, celldims=(3,4))                          == CellArrays.fill!(CuCellArray{T_Bool}(undef,2,3), falses((3,4)))
                    @test @trues(2,3, celldims=(3,4))                           == CellArrays.fill!(CuCellArray{T_Bool}(undef,2,3), trues((3,4)))
                    CUDA.allowscalar(false)
                elseif $package == $PKG_AMDGPU
                    AMDGPU.allowscalar(true)
                    @test @zeros(2,3, celldims=(3,4), eltype=Float32)           == CellArrays.fill!(ROCCellArray{T_Float32}(undef,2,3), T_Float32(zeros((3,4))))
                    @test @ones(2,3, celldims=(3,4), eltype=Float32)            == CellArrays.fill!(ROCCellArray{T_Float32}(undef,2,3), T_Float32(ones((3,4))))
                    @test typeof(@rand(2,3, celldims=(3,4), eltype=Float64))    == typeof(ROCCellArray{T_Float64,0}(undef,2,3))
                    @test typeof(@fill(9, 2,3, celldims=(3,4), eltype=Float64)) == typeof(ROCCellArray{T_Float64,0}(undef,2,3))
                    @test @falses(2,3, celldims=(3,4))                          == CellArrays.fill!(ROCCellArray{T_Bool}(undef,2,3), falses((3,4)))
                    @test @trues(2,3, celldims=(3,4))                           == CellArrays.fill!(ROCCellArray{T_Bool}(undef,2,3), trues((3,4)))
                    AMDGPU.allowscalar(false)
                else
                    @test @zeros(2,3, celldims=(3,4), eltype=Float32)           == CellArrays.fill!(CPUCellArray{T_Float32}(undef,2,3), T_Float32(zeros((3,4))))
                    @test @ones(2,3, celldims=(3,4), eltype=Float32)            == CellArrays.fill!(CPUCellArray{T_Float32}(undef,2,3), T_Float32(ones((3,4))))
                    @test typeof(@rand(2,3, celldims=(3,4), eltype=Float64))    == typeof(CPUCellArray{T_Float64,1}(undef,2,3))
                    @test typeof(@fill(9, 2,3, celldims=(3,4), eltype=Float64)) == typeof(CPUCellArray{T_Float64,1}(undef,2,3))
                    @test @falses(2,3, celldims=(3,4))                          == CellArrays.fill!(CPUCellArray{T_Bool}(undef,2,3), falses((3,4)))
                    @test @trues(2,3, celldims=(3,4))                           == CellArrays.fill!(CPUCellArray{T_Bool}(undef,2,3), trues((3,4)))
                end
            end;
            @testset "mapping to package (with celltype)" begin
                @static if $package == $PKG_CUDA
                    CUDA.allowscalar(true)
                    @test @zeros(2,3, celltype=SymmetricTensor2D)               == CellArrays.fill!(CuCellArray{SymmetricTensor2D}(undef,2,3), SymmetricTensor2D(zeros(3)))
                    @test @zeros(2,3, celltype=SymmetricTensor3D)               == CellArrays.fill!(CuCellArray{SymmetricTensor3D}(undef,2,3), SymmetricTensor3D(zeros(6)))
                    @test @zeros(2,3, celltype=Tensor2D)                        == CellArrays.fill!(CuCellArray{Tensor2D}(undef,2,3), Tensor2D(zeros((2,2,2,2))))
                    @test @zeros(2,3, celltype=SymmetricTensor2D_T{Float64})    == CellArrays.fill!(CuCellArray{SymmetricTensor2D_T{Float64}}(undef,2,3), SymmetricTensor2D_T{Float64}(zeros(3)))
                    @test @zeros(2,3, celltype=SymmetricTensor2D_Float32)       == CellArrays.fill!(CuCellArray{SymmetricTensor2D_Float32}(undef,2,3), SymmetricTensor2D_Float32(zeros(3)))
                    @test @ones(2,3, celltype=SymmetricTensor2D)                == CellArrays.fill!(CuCellArray{SymmetricTensor2D}(undef,2,3), SymmetricTensor2D(ones(3)))
                    @test typeof(@rand(2,3, celltype=SymmetricTensor2D))        == typeof(CuCellArray{SymmetricTensor2D,0}(undef,2,3))
                    @test typeof(@fill(9, 2,3, celltype=SymmetricTensor2D))     == typeof(CuCellArray{SymmetricTensor2D,0}(undef,2,3))
                    CUDA.allowscalar(false)
                elseif $package == $PKG_AMDGPU
                    AMDGPU.allowscalar(true)
                    @test @zeros(2,3, celltype=SymmetricTensor2D)               == CellArrays.fill!(ROCCellArray{SymmetricTensor2D}(undef,2,3), SymmetricTensor2D(zeros(3)))
                    @test @zeros(2,3, celltype=SymmetricTensor3D)               == CellArrays.fill!(ROCCellArray{SymmetricTensor3D}(undef,2,3), SymmetricTensor3D(zeros(6)))
                    @test @zeros(2,3, celltype=Tensor2D)                        == CellArrays.fill!(ROCCellArray{Tensor2D}(undef,2,3), Tensor2D(zeros((2,2,2,2))))
                    @test @zeros(2,3, celltype=SymmetricTensor2D_T{Float64})    == CellArrays.fill!(ROCCellArray{SymmetricTensor2D_T{Float64}}(undef,2,3), SymmetricTensor2D_T{Float64}(zeros(3)))
                    @test @zeros(2,3, celltype=SymmetricTensor2D_Float32)       == CellArrays.fill!(ROCCellArray{SymmetricTensor2D_Float32}(undef,2,3), SymmetricTensor2D_Float32(zeros(3)))
                    @test @ones(2,3, celltype=SymmetricTensor2D)                == CellArrays.fill!(ROCCellArray{SymmetricTensor2D}(undef,2,3), SymmetricTensor2D(ones(3)))
                    @test typeof(@rand(2,3, celltype=SymmetricTensor2D))        == typeof(ROCCellArray{SymmetricTensor2D,0}(undef,2,3))
                    @test typeof(@fill(9, 2,3, celltype=SymmetricTensor2D))     == typeof(ROCCellArray{SymmetricTensor2D,0}(undef,2,3))
                    AMDGPU.allowscalar(false)
                else
                    @test @zeros(2,3, celltype=SymmetricTensor2D)               == CellArrays.fill!(CPUCellArray{SymmetricTensor2D}(undef,2,3), SymmetricTensor2D(zeros(3)))
                    @test @zeros(2,3, celltype=SymmetricTensor3D)               == CellArrays.fill!(CPUCellArray{SymmetricTensor3D}(undef,2,3), SymmetricTensor3D(zeros(6)))
                    @test @zeros(2,3, celltype=Tensor2D)                        == CellArrays.fill!(CPUCellArray{Tensor2D}(undef,2,3), Tensor2D(zeros((2,2,2,2))))
                    @test @zeros(2,3, celltype=SymmetricTensor2D_T{Float64})    == CellArrays.fill!(CPUCellArray{SymmetricTensor2D_T{Float64}}(undef,2,3), SymmetricTensor2D_T{Float64}(zeros(3)))
                    @test @zeros(2,3, celltype=SymmetricTensor2D_Float32)       == CellArrays.fill!(CPUCellArray{SymmetricTensor2D_Float32}(undef,2,3), SymmetricTensor2D_Float32(zeros(3)))
                    @test @ones(2,3, celltype=SymmetricTensor2D)                == CellArrays.fill!(CPUCellArray{SymmetricTensor2D}(undef,2,3), SymmetricTensor2D(ones(3)))
                    @test typeof(@rand(2,3, celltype=SymmetricTensor2D))        == typeof(CPUCellArray{SymmetricTensor2D,1}(undef,2,3))
                    @test typeof(@fill(9, 2,3, celltype=SymmetricTensor2D))     == typeof(CPUCellArray{SymmetricTensor2D,1}(undef,2,3))
                end
            end;
            @reset_parallel_kernel()
        end;
        @testset "4. blocklength" begin
            @require !@is_initialized()
            @init_parallel_kernel($package, Float16)
            @require @is_initialized()
            T_Float16 = SMatrix{(3,4)..., Float16, prod((3,4))}
            T_Bool    = SMatrix{(3,4)..., Bool,    prod((3,4))}
            @testset "default" begin    
                @static if $package == $PKG_CUDA
                    CUDA.allowscalar(true)
                    @test typeof(  @zeros(2,3, celldims=(3,4)))                 == typeof(CuCellArray{T_Float16,0}(undef,2,3))
                    @test typeof(   @ones(2,3, celldims=(3,4)))                 == typeof(CuCellArray{T_Float16,0}(undef,2,3))
                    @test typeof(   @rand(2,3, celldims=(3,4)))                 == typeof(CuCellArray{T_Float16,0}(undef,2,3))
                    @test typeof(@fill(9, 2,3, celldims=(3,4)))                 == typeof(CuCellArray{T_Float16,0}(undef,2,3))
                    @test typeof( @falses(2,3, celldims=(3,4)))                 == typeof(CuCellArray{T_Bool,   0}(undef,2,3))
                    @test typeof(  @trues(2,3, celldims=(3,4)))                 == typeof(CuCellArray{T_Bool,   0}(undef,2,3))
                    CUDA.allowscalar(false)
                elseif $package == $PKG_AMDGPU
                    AMDGPU.allowscalar(true)
                    @test typeof(  @zeros(2,3, celldims=(3,4)))                 == typeof(ROCCellArray{T_Float16,0}(undef,2,3))
                    @test typeof(   @ones(2,3, celldims=(3,4)))                 == typeof(ROCCellArray{T_Float16,0}(undef,2,3))
                    @test typeof(   @rand(2,3, celldims=(3,4)))                 == typeof(ROCCellArray{T_Float16,0}(undef,2,3))
                    @test typeof(@fill(9, 2,3, celldims=(3,4)))                 == typeof(ROCCellArray{T_Float16,0}(undef,2,3))
                    @test typeof( @falses(2,3, celldims=(3,4)))                 == typeof(ROCCellArray{T_Bool,   0}(undef,2,3))
                    @test typeof(  @trues(2,3, celldims=(3,4)))                 == typeof(ROCCellArray{T_Bool,   0}(undef,2,3))
                    AMDGPU.allowscalar(false)
                else
                    @test typeof(  @zeros(2,3, celldims=(3,4)))                 == typeof(CPUCellArray{T_Float16,1}(undef,2,3))
                    @test typeof(   @ones(2,3, celldims=(3,4)))                 == typeof(CPUCellArray{T_Float16,1}(undef,2,3))
                    @test typeof(   @rand(2,3, celldims=(3,4)))                 == typeof(CPUCellArray{T_Float16,1}(undef,2,3))
                    @test typeof(@fill(9, 2,3, celldims=(3,4)))                 == typeof(CPUCellArray{T_Float16,1}(undef,2,3))
                    @test typeof( @falses(2,3, celldims=(3,4)))                 == typeof(CPUCellArray{T_Bool,   1}(undef,2,3))
                    @test typeof(  @trues(2,3, celldims=(3,4)))                 == typeof(CPUCellArray{T_Bool,   1}(undef,2,3))
                end
            end;
            @testset "custom" begin    
                @static if $package == $PKG_CUDA
                    CUDA.allowscalar(true)
                    @test typeof(  @zeros(2,3, celldims=(3,4), blocklength=1))  == typeof(CuCellArray{T_Float16,1}(undef,2,3))
                    @test typeof(   @ones(2,3, celldims=(3,4), blocklength=1))  == typeof(CuCellArray{T_Float16,1}(undef,2,3))
                    @test typeof(   @rand(2,3, celldims=(3,4), blocklength=1))  == typeof(CuCellArray{T_Float16,1}(undef,2,3))
                    @test typeof(@fill(9, 2,3, celldims=(3,4), blocklength=1))  == typeof(CuCellArray{T_Float16,1}(undef,2,3))
                    @test typeof( @falses(2,3, celldims=(3,4), blocklength=1))  == typeof(CuCellArray{T_Bool,   1}(undef,2,3))
                    @test typeof(  @trues(2,3, celldims=(3,4), blocklength=1))  == typeof(CuCellArray{T_Bool,   1}(undef,2,3))
                    @test typeof(  @zeros(2,3, celldims=(3,4), blocklength=3))  == typeof(CuCellArray{T_Float16,3}(undef,2,3))
                    @test typeof(   @ones(2,3, celldims=(3,4), blocklength=3))  == typeof(CuCellArray{T_Float16,3}(undef,2,3))
                    @test typeof(   @rand(2,3, celldims=(3,4), blocklength=3))  == typeof(CuCellArray{T_Float16,3}(undef,2,3))
                    @test typeof(@fill(9, 2,3, celldims=(3,4), blocklength=3))  == typeof(CuCellArray{T_Float16,3}(undef,2,3))
                    @test typeof( @falses(2,3, celldims=(3,4), blocklength=3))  == typeof(CuCellArray{T_Bool,   3}(undef,2,3))
                    @test typeof(  @trues(2,3, celldims=(3,4), blocklength=3))  == typeof(CuCellArray{T_Bool,   3}(undef,2,3))
                    CUDA.allowscalar(false)
                elseif $package == $PKG_AMDGPU
                    AMDGPU.allowscalar(true)
                    @test typeof(  @zeros(2,3, celldims=(3,4), blocklength=1))  == typeof(ROCCellArray{T_Float16,1}(undef,2,3))
                    @test typeof(   @ones(2,3, celldims=(3,4), blocklength=1))  == typeof(ROCCellArray{T_Float16,1}(undef,2,3))
                    @test typeof(   @rand(2,3, celldims=(3,4), blocklength=1))  == typeof(ROCCellArray{T_Float16,1}(undef,2,3))
                    @test typeof(@fill(9, 2,3, celldims=(3,4), blocklength=1))  == typeof(ROCCellArray{T_Float16,1}(undef,2,3))
                    @test typeof( @falses(2,3, celldims=(3,4), blocklength=1))  == typeof(ROCCellArray{T_Bool,   1}(undef,2,3))
                    @test typeof(  @trues(2,3, celldims=(3,4), blocklength=1))  == typeof(ROCCellArray{T_Bool,   1}(undef,2,3))
                    @test typeof(  @zeros(2,3, celldims=(3,4), blocklength=3))  == typeof(ROCCellArray{T_Float16,3}(undef,2,3))
                    @test typeof(   @ones(2,3, celldims=(3,4), blocklength=3))  == typeof(ROCCellArray{T_Float16,3}(undef,2,3))
                    @test typeof(   @rand(2,3, celldims=(3,4), blocklength=3))  == typeof(ROCCellArray{T_Float16,3}(undef,2,3))
                    @test typeof(@fill(9, 2,3, celldims=(3,4), blocklength=3))  == typeof(ROCCellArray{T_Float16,3}(undef,2,3))
                    @test typeof( @falses(2,3, celldims=(3,4), blocklength=3))  == typeof(ROCCellArray{T_Bool,   3}(undef,2,3))
                    @test typeof(  @trues(2,3, celldims=(3,4), blocklength=3))  == typeof(ROCCellArray{T_Bool,   3}(undef,2,3))
                    AMDGPU.allowscalar(false)
                else
                    @test typeof(  @zeros(2,3, celldims=(3,4), blocklength=0))  == typeof(CPUCellArray{T_Float16,0}(undef,2,3))
                    @test typeof(   @ones(2,3, celldims=(3,4), blocklength=0))  == typeof(CPUCellArray{T_Float16,0}(undef,2,3))
                    @test typeof(   @rand(2,3, celldims=(3,4), blocklength=0))  == typeof(CPUCellArray{T_Float16,0}(undef,2,3))
                    @test typeof(@fill(9, 2,3, celldims=(3,4), blocklength=0))  == typeof(CPUCellArray{T_Float16,0}(undef,2,3))
                    @test typeof( @falses(2,3, celldims=(3,4), blocklength=0))  == typeof(CPUCellArray{T_Bool,   0}(undef,2,3))
                    @test typeof(  @trues(2,3, celldims=(3,4), blocklength=0))  == typeof(CPUCellArray{T_Bool,   0}(undef,2,3))
                    @test typeof(  @zeros(2,3, celldims=(3,4), blocklength=3))  == typeof(CPUCellArray{T_Float16,3}(undef,2,3))
                    @test typeof(   @ones(2,3, celldims=(3,4), blocklength=3))  == typeof(CPUCellArray{T_Float16,3}(undef,2,3))
                    @test typeof(   @rand(2,3, celldims=(3,4), blocklength=3))  == typeof(CPUCellArray{T_Float16,3}(undef,2,3))
                    @test typeof(@fill(9, 2,3, celldims=(3,4), blocklength=3))  == typeof(CPUCellArray{T_Float16,3}(undef,2,3))
                    @test typeof( @falses(2,3, celldims=(3,4), blocklength=3))  == typeof(CPUCellArray{T_Bool,   3}(undef,2,3))
                    @test typeof(  @trues(2,3, celldims=(3,4), blocklength=3))  == typeof(CPUCellArray{T_Bool,   3}(undef,2,3))
                end
            end;
            @reset_parallel_kernel()
        end;
        @testset "5. Enums" begin
            @require !@is_initialized()
            @init_parallel_kernel($package, Float16)
            @require @is_initialized()
            @enum Phase air fluid solid
            T_Phase = SMatrix{(3,4)..., Phase, prod((3,4))}
            @static if $package == $PKG_CUDA
                CUDA.allowscalar(true)
                @test typeof(@rand(2,3, eltype=Phase))                                          == typeof(CUDA.CuArray(rand(Phase, 2,3)))
                @test typeof(@rand(2,3, celldims=(3,4), eltype=Phase))                          == typeof(CuCellArray{T_Phase,0}(undef,2,3))
                @test typeof(@fill(solid, 2,3, eltype=Phase))                                   == typeof(CUDA.CuArray(rand(Phase, 2,3)))
                @test typeof(@fill(solid, 2,3, celldims=(3,4), eltype=Phase))                   == typeof(CuCellArray{T_Phase,0}(undef,2,3))
                @test typeof(@fill(@rand(3,4,eltype=Phase), 2,3, celldims=(3,4), eltype=Phase)) == typeof(CuCellArray{T_Phase,0}(undef,2,3))
                CUDA.allowscalar(false)
            elseif $package == $PKG_AMDGPU
                AMDGPU.allowscalar(true)
                @test typeof(@rand(2,3, eltype=Phase))                                          == typeof(AMDGPU.ROCArray(rand(Phase, 2,3)))
                @test typeof(@rand(2,3, celldims=(3,4), eltype=Phase))                          == typeof(ROCCellArray{T_Phase,0}(undef,2,3))
                @test typeof(@fill(solid, 2,3, eltype=Phase))                                   == typeof(AMDGPU.ROCArray(rand(Phase, 2,3)))
                @test typeof(@fill(solid, 2,3, celldims=(3,4), eltype=Phase))                   == typeof(ROCCellArray{T_Phase,0}(undef,2,3))
                @test typeof(@fill(@rand(3,4,eltype=Phase), 2,3, celldims=(3,4), eltype=Phase)) == typeof(ROCCellArray{T_Phase,0}(undef,2,3))
                AMDGPU.allowscalar(false)
            else
                @test typeof(@rand(2,3, eltype=Phase))                                          == typeof(rand(Phase, 2,3))
                @test typeof(@rand(2,3, celldims=(3,4), eltype=Phase))                          == typeof(CPUCellArray{T_Phase,1}(undef,2,3))
                @test typeof(@fill(solid, 2,3, eltype=Phase))                                   == typeof(fill(solid, 2,3))
                @test typeof(@fill(solid, 2,3, celldims=(3,4), eltype=Phase))                   == typeof(CPUCellArray{T_Phase,1}(undef,2,3))
                @test typeof(@fill(@rand(3,4,eltype=Phase), 2,3, celldims=(3,4), eltype=Phase)) == typeof(CPUCellArray{T_Phase,1}(undef,2,3))
            end
            @reset_parallel_kernel()
        end;
        @testset "6. Exceptions" begin
            @require !@is_initialized()
            @init_parallel_kernel(package = $package)
            @require @is_initialized
            @testset "arguments @CellType" begin
                @test_throws ArgumentError checkargs_CellType();                                                                       # Error: isempty(args)
                @test_throws ArgumentError checkargs_CellType(:SymmetricTensor2D, :(xx, yy, zz));                                      # Error: length(posargs) != 1
                @test_throws ArgumentError checkargs_CellType(:SymmetricTensor2D);                                                     # Error: length(kwargs_expr) < 1
                @test_throws ArgumentError checkargs_CellType(:SymmetricTensor2D, :(eltype=Float32), :(fieldnames=(xx, zz, xz)), :(dims=(2,3)), :(parametric=true), :(fifthkwarg="something"));  # Error: length(kwargs_expr) > 4
                @test_throws ArgumentError _CellType(:SymmetricTensor2D, eltype=Float32, dims=:((2,3)))                                # Error: isnothing(fieldnames)
                @test_throws ArgumentError _CellType(:SymmetricTensor2D, fieldnames=:((xx, zz, xz)), dims=:((2,3)))                    # Error: isnothing(eltype) && (!parametric && eltype == NUMBERTYPE_NONE)
                @test_throws ArgumentError _CellType(:SymmetricTensor2D, fieldnames=:((xx, zz, xz)), eltype=Float32, parametric=true)  # Error: !isnothing(fieldnames) && parametric
            end;
            @reset_parallel_kernel()
        end;
    end;
)) end == nothing || true;
