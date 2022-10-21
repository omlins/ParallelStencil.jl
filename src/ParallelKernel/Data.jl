const DATA_DOC = """
    Module Data

The module Data is created in the module where `@init_parallel_kernel` is called from. It provides the following types:

--------------------------------------------------------------------------------
    Data.Number

The type of numbers used by @zeros, @ones, @rand and @fill and in all array types of module `Data` (selected with argument `numbertype` of [`@init_parallel_kernel`](@ref)).

--------------------------------------------------------------------------------
    Data.Array{ndims}

Expands to `Data.Array{numbertype, ndims}`, where `numbertype` is the datatype selected with [`@init_parallel_kernel`](@ref) and the datatype `Data.Array` is chosen to be compatible with the package for parallelization selected with [`@init_parallel_kernel`](@ref) (Array for Threads and CUDA.CuArray or CUDA.CuDeviceArray for CUDA; [`@parallel`](@ref) and [`@parallel_indices`](@ref) convert CUDA.CuArray automatically to CUDA.CuDeviceArray in kernels when required).

--------------------------------------------------------------------------------
    Data.CellArray{ndims}

Expands to `Data.CellArray{numbertype, ndims}`, where `numbertype` is the datatype selected with [`@init_parallel_kernel`](@ref) and the datatype `Data.CellArray` is chosen to be compatible with the package for parallelization selected with [`@init_parallel_kernel`](@ref) (CPUCellArray for Threads and CuCellArray or CuDeviceCellArray for CUDA; [`@parallel`](@ref) and [`@parallel_indices`](@ref) convert CUDA.CuArray automatically to CUDA.CuDeviceArray in kernels when required).

--------------------------------------------------------------------------------
    Data.Cell{S}

Expands to `Union{StaticArrays.SArray{S, numbertype}, StaticArrays.FieldArray{S, numbertype}}`, where `numbertype` is the datatype selected with [`@init_parallel_kernel`](@ref).

--------------------------------------------------------------------------------
!!! note "Advanced"
        Data.DeviceArray{ndims}

    Expands to `Data.DeviceArray{numbertype, ndims}`, where `numbertype` is the datatype selected with [`@init_parallel_kernel`](@ref) and the datatype `Data.DeviceArray` is chosen to be compatible with the package for parallelization selected with [`@init_parallel_kernel`](@ref) (Array for Threads and CUDA.CuDeviceArray for CUDA).

    !!! warning
        This datatype is not intended for explicit manual usage. [`@parallel`](@ref) and [`@parallel_indices`](@ref) convert CUDA.CuArray automatically to CUDA.CuDeviceArray in kernels when required.

    --------------------------------------------------------------------------------
        Data.DeviceCellArray{ndims}

    Expands to `Data.DeviceCellArray{numbertype, ndims}`, where `numbertype` is the datatype selected with [`@init_parallel_kernel`](@ref) and the datatype `Data.DeviceCellArray` is chosen to be compatible with the package for parallelization selected with [`@init_parallel_kernel`](@ref) (CPUCellArray for Threads and CuDeviceCellArray for CUDA).

    !!! warning
        This datatype is not intended for explicit manual usage. [`@parallel`](@ref) and [`@parallel_indices`](@ref) convert CUDA.CuArray automatically to CUDA.CuDeviceArray in kernels when required.
"""

const DATA_DOC_NUMBERTYPE_NONE = """
    Module Data

The module Data is created in the module where `@init_parallel_kernel` is called from. It provides the following types:

--------------------------------------------------------------------------------
    Data.Array{numbertype, ndims}

The datatype `Data.Array` is automatically chosen to be compatible with the package for parallelization selected with [`@init_parallel_kernel`](@ref) (Array for Threads and CUDA.CuArray or CUDA.CuDeviceArray for CUDA; [`@parallel`](@ref) and [`@parallel_indices`](@ref) convert CUDA.CuArray automatically to CUDA.CuDeviceArray in kernels when required).

--------------------------------------------------------------------------------
    Data.CellArray{numbertype, ndims}

The datatype `Data.CellArray` is automatically chosen to be compatible with the package for parallelization selected with [`@init_parallel_kernel`](@ref) (CPUCellArray for Threads and CuCellArray or CuDeviceCellArray for CUDA; [`@parallel`](@ref) and [`@parallel_indices`](@ref) convert CUDA.CuArray automatically to CUDA.CuDeviceArray in kernels when required).

--------------------------------------------------------------------------------
    Data.Cell{numbertype, S}

Expands to `Union{StaticArrays.SArray{S, numbertype}, StaticArrays.FieldArray{S, numbertype}}`, where `numbertype` is the datatype selected with [`@init_parallel_kernel`](@ref).

--------------------------------------------------------------------------------
!!! note "Advanced"
        Data.DeviceArray{numbertype, ndims}

    The datatype `Data.DeviceArray` is automatically chosen to be compatible with the package for parallelization selected with [`@init_parallel_kernel`](@ref) (Array for Threads and CUDA.CuDeviceArray for CUDA).

    !!! warning
        This datatype is not intended for explicit manual usage. [`@parallel`](@ref) and [`@parallel_indices`](@ref) convert CUDA.CuArray automatically to CUDA.CuDeviceArray in kernels when required.

    --------------------------------------------------------------------------------
        Data.DeviceCellArray{numbertype, ndims}

    The datatype `Data.DeviceCellArray` is automatically chosen to be compatible with the package for parallelization selected with [`@init_parallel_kernel`](@ref) (CPUCellArray for Threads and CuDeviceCellArray for CUDA).

    !!! warning
        This datatype is not intended for explicit manual usage. [`@parallel`](@ref) and [`@parallel_indices`](@ref) convert CUDA.CuArray automatically to CUDA.CuDeviceArray in kernels when required.
"""

function Data_cuda(numbertype::DataType)
    if numbertype == NUMBERTYPE_NONE
        :(baremodule Data # NOTE: there cannot be any newline before 'module Data' or it will create a begin end block and the module creation will fail.
            import CUDA, CellArrays, StaticArrays
            Array{T, N}                   = CUDA.CuArray{T, N}
            DeviceArray{T, N}             = CUDA.CuDeviceArray{T, N}
            Cell{T, S}                    = Union{StaticArrays.SArray{S, T}, StaticArrays.FieldArray{S, T}}
            DeviceCell{T, S}              = Union{StaticArrays.SArray{S, T}, StaticArrays.FieldArray{S, T}}
            CellArray{T_elem, N, B}       = CellArrays.CuCellArray{<:Cell{T_elem},N,B,T_elem}
            DeviceCellArray{T_elem, N, B} = CellArrays.CellArray{<:DeviceCell{T_elem},N,B,<:CUDA.CuDeviceArray{T_elem,CellArrays._N}}
        end)
    else
        :(baremodule Data # NOTE: there cannot be any newline before 'module Data' or it will create a begin end block and the module creation will fail.
            import CUDA, CellArrays, StaticArrays
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

function Data_threads(numbertype::DataType)
    if numbertype == NUMBERTYPE_NONE
        :(baremodule Data # NOTE: there cannot be any newline before 'module Data' or it will create a begin end block and the module creation will fail.
            import Base, CellArrays, StaticArrays
            Array{T, N}                    = Base.Array{T, N}
            DeviceArray{T, N}              = Base.Array{T, N}
            Cell{T, S}                     = Union{StaticArrays.SArray{S, T}, StaticArrays.FieldArray{S, T}}
            DeviceCell{T, S}               = Union{StaticArrays.SArray{S, T}, StaticArrays.FieldArray{S, T}}
            CellArray{T_elem, N, B}        = CellArrays.CPUCellArray{<:Cell{T_elem},N,B,T_elem}
            DeviceCellArray{T_elem, N, B}  = CellArrays.CPUCellArray{<:DeviceCell{T_elem},N,B,T_elem}
        end)
    else
        :(baremodule Data # NOTE: there cannot be any newline before 'module Data' or it will create a begin end block and the module creation will fail.
            import Base, CellArrays, StaticArrays
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

function Data_none()
    :(baremodule Data # NOTE: there cannot be any newline before 'module Data' or it will create a begin end block and the module creation will fail.
    end)
end
