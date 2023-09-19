const DATA_DOC = """
    Module Data

The module Data is created in the module where `@init_parallel_kernel` is called from. It provides the following types:

--------------------------------------------------------------------------------
    Data.Number

The type of numbers used by @zeros, @ones, @rand and @fill and in all array types of module `Data` (selected with argument `numbertype` of [`@init_parallel_kernel`](@ref)).

--------------------------------------------------------------------------------
    Data.Array{ndims}

Expands to `Data.Array{numbertype, ndims}`, where `numbertype` is the datatype selected with [`@init_parallel_kernel`](@ref) and the datatype `Data.Array` is chosen to be compatible with the package for parallelization selected with [`@init_parallel_kernel`](@ref) (Array for Threads, CUDA.CuArray or CUDA.CuDeviceArray for CUDA and AMDGPU.ROCArray or AMDGPU.ROCDeviceArray for AMDGPU; [`@parallel`](@ref) and [`@parallel_indices`](@ref) convert CUDA.CuArray and AMDGPU.ROCArray automatically to CUDA.CuDeviceArray and AMDGPU.ROCDeviceArray in kernels when required).

--------------------------------------------------------------------------------
    Data.CellArray{ndims}

Expands to `Data.CellArray{numbertype, ndims}`, where `numbertype` is the datatype selected with [`@init_parallel_kernel`](@ref) and the datatype `Data.CellArray` is chosen to be compatible with the package for parallelization selected with [`@init_parallel_kernel`](@ref) (CPUCellArray for Threads, CuCellArray or CuDeviceCellArray for CUDA and ROCCellArray or ROCDeviceCellArray for AMDGPU; [`@parallel`](@ref) and [`@parallel_indices`](@ref) convert CellArray automatically to DeviceCellArray when required).

--------------------------------------------------------------------------------
    Data.Cell{S}

Expands to `Union{StaticArrays.SArray{S, numbertype}, StaticArrays.FieldArray{S, numbertype}}`, where `numbertype` is the datatype selected with [`@init_parallel_kernel`](@ref).

--------------------------------------------------------------------------------
!!! note "Advanced"
        Data.DeviceArray{ndims}

    Expands to `Data.DeviceArray{numbertype, ndims}`, where `numbertype` is the datatype selected with [`@init_parallel_kernel`](@ref) and the datatype `Data.DeviceArray` is chosen to be compatible with the package for parallelization selected with [`@init_parallel_kernel`](@ref) (Array for Threads, CUDA.CuDeviceArray for CUDA AMDGPU.ROCDeviceArray for AMDGPU).

    !!! warning
        This datatype is not intended for explicit manual usage. [`@parallel`](@ref) and [`@parallel_indices`](@ref) convert CUDA.CuArray and AMDGPU.ROCArray automatically to CUDA.CuDeviceArray and AMDGPU.ROCDeviceArray in kernels when required.

    --------------------------------------------------------------------------------
        Data.DeviceCellArray{ndims}

    Expands to `Data.DeviceCellArray{numbertype, ndims}`, where `numbertype` is the datatype selected with [`@init_parallel_kernel`](@ref) and the datatype `Data.DeviceCellArray` is chosen to be compatible with the package for parallelization selected with [`@init_parallel_kernel`](@ref) (CPUCellArray for Threads, CuDeviceCellArray for CUDA and ROCDeviceCellArray for AMDGPU).

    !!! warning
        This datatype is not intended for explicit manual usage. [`@parallel`](@ref) and [`@parallel_indices`](@ref) convert CUDA.CuArray and AMDGPU.ROCArray automatically to CUDA.CuDeviceArray and AMDGPU.ROCDeviceArray in kernels when required.
"""

const DATA_DOC_NUMBERTYPE_NONE = """
    Module Data

The module Data is created in the module where `@init_parallel_kernel` is called from. It provides the following types:

--------------------------------------------------------------------------------
    Data.Array{numbertype, ndims}

The datatype `Data.Array` is automatically chosen to be compatible with the package for parallelization selected with [`@init_parallel_kernel`](@ref) (Array for Threads, CUDA.CuArray or CUDA.CuDeviceArray for CUDA and AMDGPU.ROCArray or AMDGPU.ROCDeviceArray for AMDGPU; [`@parallel`](@ref) and [`@parallel_indices`](@ref) convert CUDA.CuArray and AMDGPU.ROCArray automatically to CUDA.CuDeviceArray and AMDGPU.ROCDeviceArray in kernels when required).

--------------------------------------------------------------------------------
    Data.CellArray{numbertype, ndims}

The datatype `Data.CellArray` is automatically chosen to be compatible with the package for parallelization selected with [`@init_parallel_kernel`](@ref) (CPUCellArray for Threads, CuCellArray or CuDeviceCellArray for CUDA and ROCCellArray or ROCDeviceCellArray for AMDGPU; [`@parallel`](@ref) and [`@parallel_indices`](@ref) convert CellArray automatically to DeviceCellArray in kernels when required).

--------------------------------------------------------------------------------
    Data.Cell{numbertype, S}

Expands to `Union{StaticArrays.SArray{S, numbertype}, StaticArrays.FieldArray{S, numbertype}}`, where `numbertype` is the datatype selected with [`@init_parallel_kernel`](@ref).

--------------------------------------------------------------------------------
!!! note "Advanced"
        Data.DeviceArray{numbertype, ndims}

    The datatype `Data.DeviceArray` is automatically chosen to be compatible with the package for parallelization selected with [`@init_parallel_kernel`](@ref) (Array for Threads, CUDA.CuDeviceArray for CUDA and AMDGPU.ROCDeviceArray for AMDGPU).

    !!! warning
        This datatype is not intended for explicit manual usage. [`@parallel`](@ref) and [`@parallel_indices`](@ref) convert CUDA.CuArray and AMDGPU.ROCArray automatically to CUDA.CuDeviceArray and AMDGPU.ROCDeviceArray in kernels when required.

    --------------------------------------------------------------------------------
        Data.DeviceCellArray{numbertype, ndims}

    The datatype `Data.DeviceCellArray` is automatically chosen to be compatible with the package for parallelization selected with [`@init_parallel_kernel`](@ref) (CPUCellArray for Threads, CuDeviceCellArray for CUDA and ROCDeviceCellArray for AMDGPU).

    !!! warning
        This datatype is not intended for explicit manual usage. [`@parallel`](@ref) and [`@parallel_indices`](@ref) convert CUDA.CuArray and AMDGPU.ROCArray automatically to CUDA.CuDeviceArray and AMDGPU.ROCDeviceArray in kernels when required.
"""

function Data_cuda(numbertype::DataType)
    if numbertype == NUMBERTYPE_NONE
        :(baremodule Data # NOTE: there cannot be any newline before 'module Data' or it will create a begin end block and the module creation will fail.
            import ParallelStencil.ParallelKernel.CUDA, ParallelStencil.ParallelKernel.CellArrays, ParallelStencil.ParallelKernel.StaticArrays
            Array{T, N}                   = CUDA.CuArray{T, N}
            DeviceArray{T, N}             = CUDA.CuDeviceArray{T, N}
            Cell{T, S}                    = Union{StaticArrays.SArray{S, T}, StaticArrays.FieldArray{S, T}}
            DeviceCell{T, S}              = Union{StaticArrays.SArray{S, T}, StaticArrays.FieldArray{S, T}}
            CellArray{T_elem, N, B}       = CellArrays.CuCellArray{<:Cell{T_elem},N,B,T_elem}
            DeviceCellArray{T_elem, N, B} = CellArrays.CellArray{<:DeviceCell{T_elem},N,B,<:CUDA.CuDeviceArray{T_elem,CellArrays._N}}
        end)
    else
        :(baremodule Data # NOTE: there cannot be any newline before 'module Data' or it will create a begin end block and the module creation will fail.
            import ParallelStencil.ParallelKernel.CUDA, ParallelStencil.ParallelKernel.CellArrays, ParallelStencil.ParallelKernel.StaticArrays
            Number                         = $numbertype
            Array{N}                       = CUDA.CuArray{$numbertype, N}
            DeviceArray{N}                 = CUDA.CuDeviceArray{$numbertype, N}
            Cell{S}                        = Union{StaticArrays.SArray{S, $numbertype}, StaticArrays.FieldArray{S, $numbertype}}
            DeviceCell{S}                  = Union{StaticArrays.SArray{S, $numbertype}, StaticArrays.FieldArray{S, $numbertype}}
            CellArray{N, B}                = CellArrays.CuCellArray{<:Cell,N,B,$numbertype}
            DeviceCellArray{N, B}          = CellArrays.CellArray{<:DeviceCell,N,B,<:CUDA.CuDeviceArray{$numbertype,CellArrays._N}}
            TArray{T, N}                   = CUDA.CuArray{T, N}
            DeviceTArray{T, N}             = CUDA.CuDeviceArray{T, N}
            TCell{T, S}                    = Union{StaticArrays.SArray{S, T}, StaticArrays.FieldArray{S, T}}
            DeviceTCell{T, S}              = Union{StaticArrays.SArray{S, T}, StaticArrays.FieldArray{S, T}}
            TCellArray{T_elem, N, B}       = CellArrays.CuCellArray{<:TCell{T_elem},N,B,T_elem}
            DeviceTCellArray{T_elem, N, B} = CellArrays.CellArray{<:DeviceTCell{T_elem},N,B,<:CUDA.CuDeviceArray{T_elem,CellArrays._N}}
        end)
    end
end

function Data_amdgpu(numbertype::DataType)
    if numbertype == NUMBERTYPE_NONE
        :(baremodule Data # NOTE: there cannot be any newline before 'module Data' or it will create a begin end block and the module creation will fail.
            import ParallelStencil.ParallelKernel.AMDGPU, ParallelStencil.ParallelKernel.CellArrays, ParallelStencil.ParallelKernel.StaticArrays
            Array{T, N}                   = AMDGPU.ROCArray{T, N}
            DeviceArray{T, N}             = AMDGPU.ROCDeviceArray{T, N}
            Cell{T, S}                    = Union{StaticArrays.SArray{S, T}, StaticArrays.FieldArray{S, T}}
            DeviceCell{T, S}              = Union{StaticArrays.SArray{S, T}, StaticArrays.FieldArray{S, T}}
            CellArray{T_elem, N, B}       = CellArrays.ROCCellArray{<:Cell{T_elem},N,B,T_elem}
            DeviceCellArray{T_elem, N, B} = CellArrays.CellArray{<:DeviceCell{T_elem},N,B,<:AMDGPU.ROCDeviceArray{T_elem,CellArrays._N}}
        end)
    else
        :(baremodule Data # NOTE: there cannot be any newline before 'module Data' or it will create a begin end block and the module creation will fail.
            import ParallelStencil.ParallelKernel.AMDGPU, ParallelStencil.ParallelKernel.CellArrays, ParallelStencil.ParallelKernel.StaticArrays
            Number                         = $numbertype
            Array{N}                       = AMDGPU.ROCArray{$numbertype, N}
            DeviceArray{N}                 = AMDGPU.ROCDeviceArray{$numbertype, N}
            Cell{S}                        = Union{StaticArrays.SArray{S, $numbertype}, StaticArrays.FieldArray{S, $numbertype}}
            DeviceCell{S}                  = Union{StaticArrays.SArray{S, $numbertype}, StaticArrays.FieldArray{S, $numbertype}}
            CellArray{N, B}                = CellArrays.ROCCellArray{<:Cell,N,B,$numbertype}
            DeviceCellArray{N, B}          = CellArrays.CellArray{<:DeviceCell,N,B,<:AMDGPU.ROCDeviceArray{$numbertype,CellArrays._N}}
            TArray{T, N}                   = AMDGPU.ROCArray{T, N}
            DeviceTArray{T, N}             = AMDGPU.ROCDeviceArray{T, N}
            TCell{T, S}                    = Union{StaticArrays.SArray{S, T}, StaticArrays.FieldArray{S, T}}
            DeviceTCell{T, S}              = Union{StaticArrays.SArray{S, T}, StaticArrays.FieldArray{S, T}}
            TCellArray{T_elem, N, B}       = CellArrays.ROCCellArray{<:TCell{T_elem},N,B,T_elem}
            DeviceTCellArray{T_elem, N, B} = CellArrays.CellArray{<:DeviceTCell{T_elem},N,B,<:AMDGPU.ROCDeviceArray{T_elem,CellArrays._N}}
        end)
    end
end

function Data_threads(numbertype::DataType)
    if numbertype == NUMBERTYPE_NONE
        :(baremodule Data # NOTE: there cannot be any newline before 'module Data' or it will create a begin end block and the module creation will fail.
            import Base, ParallelStencil.ParallelKernel.CellArrays, ParallelStencil.ParallelKernel.StaticArrays
            Array{T, N}                    = Base.Array{T, N}
            DeviceArray{T, N}              = Base.Array{T, N}
            Cell{T, S}                     = Union{StaticArrays.SArray{S, T}, StaticArrays.FieldArray{S, T}}
            DeviceCell{T, S}               = Union{StaticArrays.SArray{S, T}, StaticArrays.FieldArray{S, T}}
            CellArray{T_elem, N, B}        = CellArrays.CPUCellArray{<:Cell{T_elem},N,B,T_elem}
            DeviceCellArray{T_elem, N, B}  = CellArrays.CPUCellArray{<:DeviceCell{T_elem},N,B,T_elem}
        end)
    else
        :(baremodule Data # NOTE: there cannot be any newline before 'module Data' or it will create a begin end block and the module creation will fail.
            import Base, ParallelStencil.ParallelKernel.CellArrays, ParallelStencil.ParallelKernel.StaticArrays
            Number                         = $numbertype
            Array{N}                       = Base.Array{$numbertype, N}
            DeviceArray{N}                 = Base.Array{$numbertype, N}
            Cell{S}                        = Union{StaticArrays.SArray{S, $numbertype}, StaticArrays.FieldArray{S, $numbertype}}
            DeviceCell{S}                  = Union{StaticArrays.SArray{S, $numbertype}, StaticArrays.FieldArray{S, $numbertype}}
            CellArray{N, B}                = CellArrays.CPUCellArray{<:Cell,N,B,$numbertype}
            DeviceCellArray{N, B}          = CellArrays.CPUCellArray{<:DeviceCell,N,B,$numbertype}
            TArray{T, N}                   = Base.Array{T, N}
            DeviceTArray{T, N}             = Base.Array{T, N}
            TCell{T, S}                    = Union{StaticArrays.SArray{S, T}, StaticArrays.FieldArray{S, T}}
            DeviceTCell{T, S}              = Union{StaticArrays.SArray{S, T}, StaticArrays.FieldArray{S, T}}
            TCellArray{T_elem, N, B}       = CellArrays.CPUCellArray{<:TCell{T_elem},N,B,T_elem}
            DeviceTCellArray{T_elem, N, B} = CellArrays.CPUCellArray{<:DeviceTCell{T_elem},N,B,T_elem}
        end)
    end
end

function Data_shared(numbertype::DataType)
    if numbertype == NUMBERTYPE_NONE
        quote
            NumberTuple{N_tuple, T}                                  = NTuple{N_tuple, T}
            ArrayTuple{N_tuple, T, N}                                = NTuple{N_tuple, Array{T, N}}
            DeviceArrayTuple{N_tuple, T, N}                          = NTuple{N_tuple, DeviceArray{T, N}}
            CellTuple{N_tuple, T, S}                                 = NTuple{N_tuple, Cell{T, S}}
            DeviceCellTuple{N_tuple, T, S}                           = NTuple{N_tuple, DeviceCell{T, S}}
            CellArrayTuple{N_tuple, T_elem, N, B}                    = NTuple{N_tuple, CellArray{T_elem, N, B}}
            DeviceCellArrayTuple{N_tuple, T_elem, N, B}              = NTuple{N_tuple, DeviceCellArray{T_elem, N, B}}
            NamedNumberTuple{N_tuple, T, names}                      = NamedTuple{names, NumberTuple{N_tuple, T}}
            NamedArrayTuple{N_tuple, T, N, names}                    = NamedTuple{names, ArrayTuple{N_tuple, T, N}}
            NamedDeviceArrayTuple{N_tuple, T, N, names}              = NamedTuple{names, DeviceArrayTuple{N_tuple, T, N}}
            NamedCellTuple{N_tuple, T, S, names}                     = NamedTuple{names, CellTuple{N_tuple, T, S}}
            NamedDeviceCellTuple{N_tuple, T, S, names}               = NamedTuple{names, DeviceCellTuple{N_tuple, T, S}}
            NamedCellArrayTuple{N_tuple, T_elem, N, B, names}        = NamedTuple{names, CellArrayTuple{N_tuple, T_elem, N, B}}
            NamedDeviceCellArrayTuple{N_tuple, T_elem, N, B, names}  = NamedTuple{names, DeviceCellArrayTuple{N_tuple, T_elem, N, B}}
            NumberCollection{N_tuple, T}                             = Union{NumberTuple{N_tuple, T}, NamedNumberTuple{N_tuple, T}}
            ArrayCollection{N_tuple, T, N}                           = Union{ArrayTuple{N_tuple, T, N}, NamedArrayTuple{N_tuple, T, N}}
            DeviceArrayCollection{N_tuple, T, N}                     = Union{DeviceArrayTuple{N_tuple, T, N}, NamedDeviceArrayTuple{N_tuple, T, N}}
            CellCollection{N_tuple, T, S}                            = Union{CellTuple{N_tuple, T, S}, NamedCellTuple{N_tuple, T, S}}
            DeviceCellCollection{N_tuple, T, S}                      = Union{DeviceCellTuple{N_tuple, T, S}, NamedDeviceCellTuple{N_tuple, T, S}}
            CellArrayCollection{N_tuple, T_elem, N, B}               = Union{CellArrayTuple{N_tuple, T_elem, N, B}, NamedCellArrayTuple{N_tuple, T_elem, N, B}}
            DeviceCellArrayCollection{N_tuple, T_elem, N, B}         = Union{DeviceCellArrayTuple{N_tuple, T_elem, N, B}, NamedDeviceCellArrayTuple{N_tuple, T_elem, N, B}}
        end
    else
        quote
            NumberTuple{N_tuple}                                     = NTuple{N_tuple, Number}
            ArrayTuple{N_tuple, N}                                   = NTuple{N_tuple, Array{N}}
            DeviceArrayTuple{N_tuple, N}                             = NTuple{N_tuple, DeviceArray{N}}
            CellTuple{N_tuple, S}                                    = NTuple{N_tuple, Cell{S}}
            DeviceCellTuple{N_tuple, S}                              = NTuple{N_tuple, DeviceCell{S}}
            CellArrayTuple{N_tuple, N, B}                            = NTuple{N_tuple, CellArray{N, B}}
            DeviceCellArrayTuple{N_tuple, N, B}                      = NTuple{N_tuple, DeviceCellArray{N, B}}
            TNumberTuple{N_tuple, T}                                 = NTuple{N_tuple, T}
            TArrayTuple{N_tuple, T, N}                               = NTuple{N_tuple, TArray{T, N}}
            DeviceTArrayTuple{N_tuple, T, N}                         = NTuple{N_tuple, DeviceTArray{T, N}}
            TCellTuple{N_tuple, T, S}                                = NTuple{N_tuple, TCell{T, S}}
            DeviceTCellTuple{N_tuple, T, S}                          = NTuple{N_tuple, DeviceTCell{T, S}}
            TCellArrayTuple{N_tuple, T_elem, N, B}                   = NTuple{N_tuple, TCellArray{T_elem, N, B}}
            DeviceTCellArrayTuple{N_tuple, T_elem, N, B}             = NTuple{N_tuple, DeviceTCellArray{T_elem, N, B}}
            NamedNumberTuple{N_tuple, names}                         = NamedTuple{names, NumberTuple{N_tuple}}
            NamedArrayTuple{N_tuple, N, names}                       = NamedTuple{names, ArrayTuple{N_tuple, N}}
            NamedDeviceArrayTuple{N_tuple, N, names}                 = NamedTuple{names, DeviceArrayTuple{N_tuple, N}}
            NamedCellTuple{N_tuple, S, names}                        = NamedTuple{names, CellTuple{N_tuple, S}}
            NamedDeviceCellTuple{N_tuple, S, names}                  = NamedTuple{names, DeviceCellTuple{N_tuple, S}}
            NamedCellArrayTuple{N_tuple, N, B, names}                = NamedTuple{names, CellArrayTuple{N_tuple, N, B}}
            NamedDeviceCellArrayTuple{N_tuple, N, B, names}          = NamedTuple{names, DeviceCellArrayTuple{N_tuple, N, B}}
            NamedTNumberTuple{N_tuple, T, names}                     = NamedTuple{names, TNumberTuple{N_tuple, T}}
            NamedTArrayTuple{N_tuple, T, N, names}                   = NamedTuple{names, TArrayTuple{N_tuple, T, N}}
            NamedDeviceTArrayTuple{N_tuple, T, N, names}             = NamedTuple{names, DeviceTArrayTuple{N_tuple, T, N}}
            NamedTCellTuple{N_tuple, T, S, names}                    = NamedTuple{names, TCellTuple{N_tuple, T, S}}
            NamedDeviceTCellTuple{N_tuple, T, S, names}              = NamedTuple{names, DeviceTCellTuple{N_tuple, T, S}}
            NamedTCellArrayTuple{N_tuple, T_elem, N, B, names}       = NamedTuple{names, TCellArrayTuple{N_tuple, T_elem, N, B}}
            NamedDeviceTCellArrayTuple{N_tuple, T_elem, N, B, names} = NamedTuple{names, DeviceTCellArrayTuple{N_tuple, T_elem, N, B}}
            NumberCollection{N_tuple}                                = Union{NumberTuple{N_tuple}, NamedNumberTuple{N_tuple}}
            ArrayCollection{N_tuple, N}                              = Union{ArrayTuple{N_tuple, N}, NamedArrayTuple{N_tuple, N}}
            DeviceArrayCollection{N_tuple, N}                        = Union{DeviceArrayTuple{N_tuple, N}, NamedDeviceArrayTuple{N_tuple, N}}
            CellCollection{N_tuple, S}                               = Union{CellTuple{N_tuple, S}, NamedCellTuple{N_tuple, S}}
            DeviceCellCollection{N_tuple, S}                         = Union{DeviceCellTuple{N_tuple, S}, NamedDeviceCellTuple{N_tuple, S}}
            CellArrayCollection{N_tuple, N, B}                       = Union{CellArrayTuple{N_tuple, N, B}, NamedCellArrayTuple{N_tuple, N, B}}
            DeviceCellArrayCollection{N_tuple, N, B}                 = Union{DeviceCellArrayTuple{N_tuple, N, B}, NamedDeviceCellArrayTuple{N_tuple, N, B}}
            TNumberCollection{N_tuple, T}                            = Union{TNumberTuple{N_tuple, T}, NamedTNumberTuple{N_tuple, T}}
            TArrayCollection{N_tuple, T, N}                          = Union{TArrayTuple{N_tuple, T, N}, NamedTArrayTuple{N_tuple, T, N}}
            DeviceTArrayCollection{N_tuple, T, N}                    = Union{DeviceTArrayTuple{N_tuple, T, N}, NamedDeviceTArrayTuple{N_tuple, T, N}}
            TCellCollection{N_tuple, T, S}                           = Union{TCellTuple{N_tuple, T, S}, NamedTCellTuple{N_tuple, T, S}}
            DeviceTCellCollection{N_tuple, T, S}                     = Union{DeviceTCellTuple{N_tuple, T, S}, NamedDeviceTCellTuple{N_tuple, T, S}}
            TCellArrayCollection{N_tuple, T_elem, N, B}              = Union{TCellArrayTuple{N_tuple, T_elem, N, B}, NamedTCellArrayTuple{N_tuple, T_elem, N, B}}
            DeviceTCellArrayCollection{N_tuple, T_elem, N, B}        = Union{DeviceTCellArrayTuple{N_tuple, T_elem, N, B}, NamedDeviceTCellArrayTuple{N_tuple, T_elem, N, B}}
        end
    end
end

function Data_none()
    :(baremodule Data # NOTE: there cannot be any newline before 'module Data' or it will create a begin end block and the module creation will fail.
    end)
end
