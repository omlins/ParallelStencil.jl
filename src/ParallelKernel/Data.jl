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
            import CUDA, StaticArrays
            Array{T, N}              = CUDA.CuArray{T, N}
            DeviceArray{T, N}        = CUDA.CuDeviceArray{T, N}
            Cell{T, S}               = StaticArrays.SArray{S, T}
            DeviceCell{T, S}         = StaticArrays.SArray{S, T}
            CellArray{T, N}       = CUDA.CuArray{<:Cell{T}, N}
            DeviceCellArray{T, N} = CUDA.CuDeviceArray{<:DeviceCell{T}, N}
        end)
    else
        :(baremodule Data # NOTE: there cannot be any newline before 'module Data' or it will create a begin end block and the module creation will fail.
            import CUDA, StaticArrays
            Number                    = $numbertype
            Array{N}                  = CUDA.CuArray{$numbertype, N}
            DeviceArray{N}            = CUDA.CuDeviceArray{$numbertype, N}
            Cell{S}                   = StaticArrays.SArray{S, $numbertype}
            DeviceCell{S}             = StaticArrays.SArray{S, $numbertype}
            CellArray{N}           = CUDA.CuArray{<:Cell, N}
            DeviceCellArray{N}     = CUDA.CuDeviceArray{<:DeviceCell, N}
            TArray{T, N}              = CUDA.CuArray{T, N}
            DeviceTArray{T, N}        = CUDA.CuDeviceArray{T, N}
            TCell{T, S}               = StaticArrays.SArray{S, T}
            DeviceTCell{T, S}         = StaticArrays.SArray{S, T}
            TCellArray{T, N}       = CUDA.CuArray{<:TCell{T}, N}
            DeviceTCellArray{T, N} = CUDA.CuDeviceArray{<:DeviceTCell{T}, N}
        end)
    end
end

function Data_threads(numbertype::DataType)
    if numbertype == NUMBERTYPE_NONE
        :(baremodule Data # NOTE: there cannot be any newline before 'module Data' or it will create a begin end block and the module creation will fail.
            import Base, StaticArrays
            Array{T, N}              = Base.Array{T, N}
            DeviceArray{T, N}        = Base.Array{T, N}
            Cell{T, S}               = StaticArrays.SArray{S, T}
            DeviceCell{T, S}         = StaticArrays.SArray{S, T}
            CellArray{T, N}       = Base.Array{<:Cell{T}, N}
            DeviceCellArray{T, N} = Base.Array{<:DeviceCell{T}, N}
        end)
    else
        :(baremodule Data # NOTE: there cannot be any newline before 'module Data' or it will create a begin end block and the module creation will fail.
            import Base, StaticArrays
            Number                    = $numbertype
            Array{N}                  = Base.Array{$numbertype, N}
            DeviceArray{N}            = Base.Array{$numbertype, N}
            Cell{S}                   = StaticArrays.SArray{S, $numbertype}
            DeviceCell{S}             = StaticArrays.SArray{S, $numbertype}
            CellArray{N}           = Base.Array{<:Cell, N}
            DeviceCellArray{N}     = Base.Array{<:DeviceCell, N}
            TArray{T, N}              = Base.Array{T, N}
            DeviceTArray{T, N}        = Base.Array{T, N}
            TCell{T, S}               = StaticArrays.SArray{S, T}
            DeviceTCell{T, S}         = StaticArrays.SArray{S, T}
            TCellArray{T, N}       = Base.Array{<:TCell{T}, N}
            DeviceTCellArray{T, N} = Base.Array{<:DeviceTCell{T}, N}
        end)
    end
end

function Data_none()
    :(baremodule Data # NOTE: there cannot be any newline before 'module Data' or it will create a begin end block and the module creation will fail.
    end)
end
