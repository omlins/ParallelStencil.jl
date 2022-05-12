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

#TODO

--------------------------------------------------------------------------------
!!! note "Advanced"
        Data.DeviceArray{ndims}

    Expands to `Data.DeviceArray{numbertype, ndims}`, where `numbertype` is the datatype selected with [`@init_parallel_kernel`](@ref) and the datatype `Data.DeviceArray` is chosen to be compatible with the package for parallelization selected with [`@init_parallel_kernel`](@ref) (Array for Threads and CUDA.CuDeviceArray for CUDA).

    !!! warning
        This datatype is not intended for explicit manual usage. [`@parallel`](@ref) and [`@parallel_indices`](@ref) convert CUDA.CuArray automatically to CUDA.CuDeviceArray in kernels when required.

--------------------------------------------------------------------------------
        Data.DeviceCellArray{ndims}

    #TODO
"""

function Data_cuda(numbertype::DataType)
    if numbertype == NUMBERTYPE_NONE
        :(baremodule Data # NOTE: there cannot be any newline before 'module Data' or it will create a begin end block and the module creation will fail.
            import CUDA, CellArrays, StaticArrays
            Array{T, N}                 = CUDA.CuArray{T, N}
            DeviceArray{T, N}           = CUDA.CuDeviceArray{T, N}
            Cell{T, S}                  = StaticArrays.SArray{S, T}
            DeviceCell{T, S}            = StaticArrays.SArray{S, T}
            CellArray{T_elem, N}        = CellArrays.CuCellArray{<:Cell{T_elem},N,0,T_elem} # Note: B is currently fixed to 0. This can be generalized later.
            DeviceCellArray{T_elem, N}  = CellArrays.CellArray{CUDA.CuDeviceArray,<:DeviceCell{T_elem},N,0,T_elem}
        end)
    else
        :(baremodule Data # NOTE: there cannot be any newline before 'module Data' or it will create a begin end block and the module creation will fail.
            import CUDA, CellArrays, StaticArrays
            Number                      = $numbertype
            Array{N}                    = CUDA.CuArray{$numbertype, N}
            DeviceArray{N}              = CUDA.CuDeviceArray{$numbertype, N}
            Cell{S}                     = StaticArrays.SArray{S, $numbertype}
            DeviceCell{S}               = StaticArrays.SArray{S, $numbertype}
            CellArray{N}                = CellArrays.CuCellArray{<:Cell,N,0,$numbertype}
            DeviceCellArray{N}          = CellArrays.CellArray{CUDA.CuDeviceArray,<:DeviceCell,N,0,$numbertype}
            TArray{T, N}                = CUDA.CuArray{T, N}
            DeviceTArray{T, N}          = CUDA.CuDeviceArray{T, N}
            TCell{T, S}                 = StaticArrays.SArray{S, T}
            DeviceTCell{T, S}           = StaticArrays.SArray{S, T}
            TCellArray{T_elem, N}       = CellArrays.CuCellArray{<:TCell{T_elem},N,0,T_elem}
            DeviceTCellArray{T_elem, N} = CellArrays.CellArray{CUDA.CuDeviceArray,<:DeviceTCell{T_elem},N,0,T_elem}
        end)
    end
end

function Data_threads(numbertype::DataType)
    if numbertype == NUMBERTYPE_NONE
        :(baremodule Data # NOTE: there cannot be any newline before 'module Data' or it will create a begin end block and the module creation will fail.
            import Base, CellArrays, StaticArrays
            Array{T, N}                 = Base.Array{T, N}
            DeviceArray{T, N}           = Base.Array{T, N}
            Cell{T, S}                  = StaticArrays.SArray{S, T}
            DeviceCell{T, S}            = StaticArrays.SArray{S, T}
            CellArray{T_elem, N}        = CellArrays.CPUCellArray{<:Cell{T_elem},N,0,T_elem} # Note: B is currently fixed to 0. This can be generalized later.
            DeviceCellArray{T_elem, N}  = CellArrays.CPUCellArray{<:DeviceCell{T_elem},N,0,T_elem}
        end)
    else
        :(baremodule Data # NOTE: there cannot be any newline before 'module Data' or it will create a begin end block and the module creation will fail.
            import Base, CellArrays, StaticArrays
            Number                      = $numbertype
            Array{N}                    = Base.Array{$numbertype, N}
            DeviceArray{N}              = Base.Array{$numbertype, N}
            Cell{S}                     = StaticArrays.SArray{S, $numbertype}
            DeviceCell{S}               = StaticArrays.SArray{S, $numbertype}
            CellArray{N}                = CellArrays.CPUCellArray{<:Cell,N,0,$numbertype}
            DeviceCellArray{N}          = CellArrays.CPUCellArray{<:DeviceCell,N,0,$numbertype}
            TArray{T, N}                = Base.Array{T, N}
            DeviceTArray{T, N}          = Base.Array{T, N}
            TCell{T, S}                 = StaticArrays.SArray{S, T}
            DeviceTCell{T, S}           = StaticArrays.SArray{S, T}
            TCellArray{T_elem, N}       = CellArrays.CPUCellArray{<:TCell{T_elem},N,0,T_elem}
            DeviceTCellArray{T_elem, N} = CellArrays.CPUCellArray{<:DeviceTCell{T_elem},N,0,T_elem}
        end)
    end
end

function Data_none()
    :(baremodule Data # NOTE: there cannot be any newline before 'module Data' or it will create a begin end block and the module creation will fail.
    end)
end
