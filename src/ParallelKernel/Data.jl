const DATA_DOC = """
    Module Data

The module Data is created in the module where `@init_parallel_kernel` is called from. It provides the following types:

--------------------------------------------------------------------------------
    Data.Number

The type of numbers used by @zeros, @ones and @rand and in all array types of module `Data` (selected with argument `numbertype` of [`@init_parallel_kernel`](@ref)).

--------------------------------------------------------------------------------
    Data.Array{ndims}

Expands to `Data.Array{numbertype, ndims}`, where `numbertype` is the datatype selected with [`@init_parallel_kernel`](@ref) and the datatype `Data.Array` is chosen to be compatible with the package for parallelization selected with [`@init_parallel_kernel`](@ref) (Array for Threads and CUDA.CuArray or CUDA.CuDeviceArray for CUDA; [`@parallel`](@ref) and [`@parallel_indices`](@ref) convert CUDA.CuArray automatically to CUDA.CuDeviceArray in kernels when required).

--------------------------------------------------------------------------------
!!! note "Advanced"
        Data.DeviceArray{ndims}

    Expands to `Data.DeviceArray{numbertype, ndims}`, where `numbertype` is the datatype selected with [`@init_parallel_kernel`](@ref) and the datatype `Data.DeviceArray` is chosen to be compatible with the package for parallelization selected with [`@init_parallel_kernel`](@ref) (Array for Threads and CUDA.CuDeviceArray for CUDA).

    !!! warning
        This datatype is not intended for explicit manual usage. [`@parallel`](@ref) and [`@parallel_indices`](@ref) convert CUDA.CuArray automatically to CUDA.CuDeviceArray in kernels when required.
"""

function Data_cuda(numbertype::DataType)
    :(baremodule Data # NOTE: there cannot be any newline before 'module Data' or it will create a begin end block and the module creation will fail.
        import CUDA
        Number         = $numbertype
        Array{N}       = CUDA.CuArray{$numbertype, N}
        DeviceArray{N} = CUDA.CuDeviceArray{$numbertype, N}
    end)
end

function Data_threads(numbertype::DataType)
    :(baremodule Data # NOTE: there cannot be any newline before 'module Data' or it will create a begin end block and the module creation will fail.
        import Base
        Number         = $numbertype
        Array{N}       = Base.Array{$numbertype, N}
        DeviceArray{N} = Base.Array{$numbertype, N}
    end)
end

function Data_none()
    :(baremodule Data # NOTE: there cannot be any newline before 'module Data' or it will create a begin end block and the module creation will fail.
    end)
end
