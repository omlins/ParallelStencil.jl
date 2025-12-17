const DATA_DOC = """
    Module Data

The module Data is created in the module where `@init_parallel_kernel` is called from. It provides data types, some of which are organized in submodules.

!!! note "KernelAbstractions workflows"
    `Data` / `TData` are generated only when [`@init_parallel_kernel`](@ref) selects a single-architecture backend (CUDA, AMDGPU, Metal, Threads, Polyester). KernelAbstractions omits these modules because hardware is chosen at runtime via [`select_hardware`](@ref) / [`current_hardware`](@ref); allocate arrays after selecting a runtime symbol as described in [Interactive prototyping with runtime hardware selection](@ref interactive-prototyping-with-runtime-hardware-selection).

It contains the following submodules:

    Data.Fields
!!! note "Advanced"
        Data.Device
        Data.Fields.Device

    For each datatype in Data and Data.Fields exist a corresponding datatype in Data.Device and Data.Fields.Device, respectively.

    !!! warning
        These Device datatypes are not intended for explicit manual usage. [`@parallel`](@ref) and [`@parallel_indices`](@ref) convert datatypes from Data and Data.Fields automatically to corresponding datatypes from Data.Device and Data.Fields.Device, respectively, in kernels when required.


The module Data provides the following types at the top level:

--------------------------------------------------------------------------------
    Data.Number

The type of numbers used by @zeros, @ones, @rand and @fill and in all array types of module `Data` (selected with argument `numbertype` of [`@init_parallel_kernel`](@ref)).

--------------------------------------------------------------------------------
    Data.Index

The type of indices used in parallel kernels.

--------------------------------------------------------------------------------
    Data.Array{ndims}

Expands to `Data.Array{numbertype, ndims}`, where `numbertype` is the datatype selected with [`@init_parallel_kernel`](@ref) and the datatype `Data.Array` is chosen to be compatible with the package for parallelization selected with [`@init_parallel_kernel`](@ref) (Array for Threads or Polyester, CUDA.CuArray or CUDA.CuDeviceArray for CUDA, AMDGPU.ROCArray or AMDGPU.ROCDeviceArray for AMDGPU and Metal.MtlArray or Metal.MtlDeviceArray for Metal; [`@parallel`](@ref) and [`@parallel_indices`](@ref) convert CUDA.CuArray, AMDGPU.ROCArray and Metal.MtlArray automatically to CUDA.CuDeviceArray, AMDGPU.ROCDeviceArray and Metal.MtlDeviceArray in kernels when required).

--------------------------------------------------------------------------------
    Data.CellArray{ndims}

Expands to `Data.CellArray{numbertype, ndims}`, where `numbertype` is the datatype selected with [`@init_parallel_kernel`](@ref) and the datatype `Data.CellArray` is chosen to be compatible with the package for parallelization selected with [`@init_parallel_kernel`](@ref) (CPUCellArray for Threads or Polyester, CuCellArray or CuDeviceCellArray for CUDA, ROCCellArray or ROCDeviceCellArray for AMDGPU and MtlCellArray or MtlDeviceCellArray for Metal; [`@parallel`](@ref) and [`@parallel_indices`](@ref) convert CellArray automatically to DeviceCellArray when required).

--------------------------------------------------------------------------------
    Data.Cell{S}

Expands to `Union{StaticArrays.SArray{S, numbertype}, StaticArrays.FieldArray{S, numbertype}}`, where `numbertype` is the datatype selected with [`@init_parallel_kernel`](@ref).

--------------------------------------------------------------------------------
    Data.NumberTuple{N_tuple} | Data.NamedNumberTuple{N_tuple, names} | Data.NumberCollection{N_tuple}

Expands to: `NTuple{N_tuple, Data.Number}` | `NamedTuple{names, NTuple{N_tuple, Data.Number}}` | `Union{Data.NumberTuple{N_tuple}, Data.NamedNumberTuple{N_tuple}}`

--------------------------------------------------------------------------------
    Data.IndexTuple{N_tuple} | Data.NamedIndexTuple{N_tuple, names} | Data.IndexCollection{N_tuple}

Expands to: `NTuple{N_tuple, Data.Index}` | `NamedTuple{names, NTuple{N_tuple, Data.Index}}` | `Union{Data.IndexTuple{N_tuple}, Data.NamedIndexTuple{N_tuple}}`

--------------------------------------------------------------------------------
    Data.ArrayTuple{N_tuple, N} | Data.NamedArrayTuple{N_tuple, N, names} | Data.ArrayCollection{N_tuple, N}

Expands to: `NTuple{N_tuple, Data.Array{N}}` | `NamedTuple{names, NTuple{N_tuple, Data.Array{N}}}` | `Union{Data.ArrayTuple{N_tuple, N}, Data.NamedArrayTuple{N_tuple, N}}`

--------------------------------------------------------------------------------
    Data.CellArrayTuple{N_tuple, N, B} | Data.NamedCellArrayTuple{N_tuple, N, B, names} | Data.CellArrayCollection{N_tuple, N, B}

Expands to: `NTuple{N_tuple, Data.CellArray{N, B}}` | `NamedTuple{names, NTuple{N_tuple, Data.CellArray{N, B}}}` | `Union{Data.CellArrayTuple{N_tuple, N, B}, Data.NamedCellArrayTuple{N_tuple, N, B}}`

--------------------------------------------------------------------------------
    Data.CellTuple{N_tuple, S} | Data.NamedCellTuple{N_tuple, S, names} | Data.CellCollection{N_tuple, S}

Expands to: `NTuple{N_tuple, Data.Cell{S}}` | `NamedTuple{names, NTuple{N_tuple, Data.Cell{S}}}` | `Union{Data.CellTuple{N_tuple, S}, Data.NamedCellTuple{N_tuple, S}}`

--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
    Submodule Data.Fields

The submodule Data.Fields provides the types for fields allocated with macros from ParallelKernel.FieldAllocators. 

!!! note "Usage"
    using .Data.Fields # Note the preceeding dot!

Data.Fields provides the following types:

--------------------------------------------------------------------------------
    Field

A scalar field, on a grid of size `gridsize`; allocated with `@Fields`.

--------------------------------------------------------------------------------
    {X|Y|Z}Field

A scalar field of the same size as the {X|Y|Z}-component of a `VectorField`, on a grid of size `gridsize`; allocated with `@{X|Y|Z}Field`.

--------------------------------------------------------------------------------
    B{X|Y|Z}Field

A scalar field of the same size as the {X|Y|Z}-component of a `BVectorField` (a vector field including boundaries), on a grid of size `gridsize`; allocated with `@B{X|Y|Z}Field`.

--------------------------------------------------------------------------------
    {XX|YY|ZZ|XY|XZ|YZ}Field

A scalar field of the same size as the {XX|YY|ZZ|XY|XZ|YZ}-component of a `TensorField`, on a grid of size `gridsize`; allocated with `@{XX|YY|ZZ|XY|XZ|YZ}Field`.

--------------------------------------------------------------------------------
    VectorField

A vector field, on a grid of size `gridsize`; allocated with `@VectorField`.

--------------------------------------------------------------------------------
    BVectorField

A vector field including boundaries, on a grid of size `gridsize`; allocated with `@BVectorField`.

--------------------------------------------------------------------------------
    TensorField

A tensor field, on a grid of size `gridsize`; allocated with `@TensorField`.
"""

const DATA_DOC_NUMBERTYPE_NONE = """
    Module Data

The module Data is created in the module where `@init_parallel_kernel` is called from. It provides data types, some of which are organized in submodules.

!!! note "KernelAbstractions workflows"
    `Data` / `TData` are generated only when [`@init_parallel_kernel`](@ref) selects a single-architecture backend (CUDA, AMDGPU, Metal, Threads, Polyester). KernelAbstractions omits these modules because hardware is chosen at runtime via [`select_hardware`](@ref) / [`current_hardware`](@ref); allocate arrays after selecting a runtime symbol as described in [Interactive prototyping with runtime hardware selection](@ref interactive-prototyping-with-runtime-hardware-selection).

It contains the following submodules:

    Data.Fields
!!! note "Advanced"
        Data.Device
        Data.Fields.Device

    For each datatype in Data and Data.Fields exist a corresponding datatype in Data.Device and Data.Fields.Device, respectively.

    !!! warning
        These Device datatypes are not intended for explicit manual usage. [`@parallel`](@ref) and [`@parallel_indices`](@ref) convert datatypes from Data and Data.Fields automatically to corresponding datatypes from Data.Device and Data.Fields.Device, respectively, in kernels when required.


The module Data provides the following types at the top level:

--------------------------------------------------------------------------------
    Data.Index

The type of indices used in parallel kernels.

--------------------------------------------------------------------------------
    Data.Array{numbertype, ndims}

The datatype `Data.Array` is automatically chosen to be compatible with the package for parallelization selected with [`@init_parallel_kernel`](@ref) (Array for Threads or Polyester, CUDA.CuArray or CUDA.CuDeviceArray for CUDA, AMDGPU.ROCArray or AMDGPU.ROCDeviceArray for AMDGPU and Metal.MtlArray or Metal.MtlDeviceArray for Metal; [`@parallel`](@ref) and [`@parallel_indices`](@ref) convert CUDA.CuArray, AMDGPU.ROCArray and Metal.MtlArray automatically to CUDA.CuDeviceArray, AMDGPU.ROCDeviceArray and Metal.MtlDeviceArray in kernels when required).

--------------------------------------------------------------------------------
    Data.CellArray{numbertype, ndims}

The datatype `Data.CellArray` is automatically chosen to be compatible with the package for parallelization selected with [`@init_parallel_kernel`](@ref) (CPUCellArray for Threads or Polyester, CuCellArray or CuDeviceCellArray for CUDA, ROCCellArray or ROCDeviceCellArray for AMDGPU and MtlCellArray or MetalDeviceCellArray for Metal; [`@parallel`](@ref) and [`@parallel_indices`](@ref) convert CellArray automatically to DeviceCellArray in kernels when required).

--------------------------------------------------------------------------------
    Data.Cell{numbertype, S}

Expands to `Union{StaticArrays.SArray{S, numbertype}, StaticArrays.FieldArray{S, numbertype}}`, where `numbertype` is the datatype selected with [`@init_parallel_kernel`](@ref).

--------------------------------------------------------------------------------
    Data.NumberTuple{N_tuple, numbertype} | Data.NamedNumberTuple{N_tuple, numbertype, names} | Data.NumberCollection{N_tuple, numbertype}

Expands to: `NTuple{N_tuple, numbertype}` | `NamedTuple{names, NTuple{N_tuple, numbertype}}` | `Union{Data.NumberTuple{N_tuple, numbertype}, Data.NamedNumberTuple{N_tuple, numbertype}}`

--------------------------------------------------------------------------------
    Data.IndexTuple{N_tuple} | Data.NamedIndexTuple{N_tuple, names} | Data.IndexCollection{N_tuple}

Expands to: `NTuple{N_tuple, Data.Index}` | `NamedTuple{names, NTuple{N_tuple, Data.Index}}` | `Union{Data.IndexTuple{N_tuple}, Data.NamedIndexTuple{N_tuple}}`

--------------------------------------------------------------------------------
    Data.ArrayTuple{N_tuple, numbertype, N} | Data.NamedArrayTuple{N_tuple, numbertype, N, names} | Data.ArrayCollection{N_tuple, numbertype, N}

Expands to: `NTuple{N_tuple, Data.Array{numbertype, N}}` | `NamedTuple{names, NTuple{N_tuple, Data.Array{numbertype, N}}}` | `Union{Data.ArrayTuple{N_tuple, numbertype, N}, Data.NamedArrayTuple{N_tuple, numbertype, N}}`

--------------------------------------------------------------------------------
    Data.CellArrayTuple{N_tuple, numbertype, N, B} | Data.NamedCellArrayTuple{N_tuple, numbertype, N, B, names} | Data.CellArrayCollection{N_tuple, numbertype, N, B}

Expands to: `NTuple{N_tuple, Data.CellArray{numbertype, N, B}}` | `NamedTuple{names, NTuple{N_tuple, Data.CellArray{numbertype, N, B}}}` | `Union{Data.CellArrayTuple{N_tuple, numbertype, N, B}, Data.NamedCellArrayTuple{N_tuple, numbertype, N, B}}`

--------------------------------------------------------------------------------
    Data.CellTuple{N_tuple, numbertype, S} | Data.NamedCellTuple{N_tuple, numbertype, S, names} | Data.CellCollection{N_tuple, numbertype, S}

Expands to: `NTuple{N_tuple, Data.Cell{numbertype, S}}` | `NamedTuple{names, NTuple{N_tuple, Data.Cell{numbertype, S}}}` | `Union{Data.CellTuple{N_tuple, numbertype, S}, Data.NamedCellTuple{N_tuple, numbertype, S}}`

--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
    Submodule Data.Fields

The submodule Data.Fields provides the types for fields allocated with macros from ParallelKernel.FieldAllocators. 

!!! note "Usage"
    using .Data.Fields # Note the preceeding dot!

Data.Fields provides the following types:

--------------------------------------------------------------------------------
    Field

A scalar field, on a grid of size `gridsize`; allocated with `@Fields`.

--------------------------------------------------------------------------------
    {X|Y|Z}Field

A scalar field of the same size as the {X|Y|Z}-component of a `VectorField`, on a grid of size `gridsize`; allocated with `@{X|Y|Z}Field`.

--------------------------------------------------------------------------------
    B{X|Y|Z}Field

A scalar field of the same size as the {X|Y|Z}-component of a `BVectorField` (a vector field including boundaries), on a grid of size `gridsize`; allocated with `@B{X|Y|Z}Field`.

--------------------------------------------------------------------------------
    {XX|YY|ZZ|XY|XZ|YZ}Field

A scalar field of the same size as the {XX|YY|ZZ|XY|XZ|YZ}-component of a `TensorField`, on a grid of size `gridsize`; allocated with `@{XX|YY|ZZ|XY|XZ|YZ}Field`.

--------------------------------------------------------------------------------
    VectorField

A vector field, on a grid of size `gridsize`; allocated with `@VectorField`.

--------------------------------------------------------------------------------
    BVectorField

A vector field including boundaries, on a grid of size `gridsize`; allocated with `@BVectorField`.

--------------------------------------------------------------------------------
    TensorField

A tensor field, on a grid of size `gridsize`; allocated with `@TensorField`.
"""


# EMPTY MODULES

function Data_none()
    :(baremodule Data
    end)
end

function TData_none()
    :(baremodule TData
    end)
end


# CUDA

function Data_cuda(numbertype::DataType, indextype::DataType)
    Data_module = if (numbertype == NUMBERTYPE_NONE)
        :(baremodule $MODULENAME_DATA # NOTE: there cannot be any newline before 'module Data' or it will create a begin end block and the module creation will fail.
            import Base, CUDA, ParallelStencil.ParallelKernel.CellArrays, ParallelStencil.ParallelKernel.StaticArrays
            # TODO: the constructors defined by CellArrays.@define_CuCellArray lead to pre-compilation issues due to a bug in Julia. We therefore only create the type alias here for now.
            const CuCellArray{T,N,B,T_elem} = CellArrays.CellArray{T,N,B,CUDA.CuArray{T_elem,CellArrays._N}}
            # CellArrays.@define_CuCellArray
            # export CuCellArray
            const Index                     = $indextype
            const Array{T, N}               = CUDA.CuArray{T, N}
            const Cell{T, S}                = Union{StaticArrays.SArray{S, T}, StaticArrays.FieldArray{S, T}}
            const CellArray{T_elem, N, B}   = CuCellArray{<:Cell{T_elem},N,B,T_elem}
            $(Data_xpu_exprs(numbertype)) 
            $(Data_Device_cuda(numbertype, indextype))
            $(Data_Fields(numbertype, indextype))
        end)
    else
        :(baremodule $MODULENAME_DATA
            import Base, CUDA, ParallelStencil.ParallelKernel.CellArrays, ParallelStencil.ParallelKernel.StaticArrays
            # TODO: the constructors defined by CellArrays.@define_CuCellArray lead to pre-compilation issues due to a bug in Julia. We therefore only create the type alias here for now.
            const CuCellArray{T,N,B,T_elem} = CellArrays.CellArray{T,N,B,CUDA.CuArray{T_elem,CellArrays._N}}
            # CellArrays.@define_CuCellArray
            # export CuCellArray
            const Index                     = $indextype
            const Number                    = $numbertype
            const Array{N}                  = CUDA.CuArray{$numbertype, N}
            const Cell{S}                   = Union{StaticArrays.SArray{S, $numbertype}, StaticArrays.FieldArray{S, $numbertype}}
            const CellArray{N, B}           = CuCellArray{<:Cell,N,B,$numbertype}
            $(Data_xpu_exprs(numbertype))
            $(Data_Device_cuda(numbertype, indextype))
            $(Data_Fields(numbertype, indextype))
        end)
    end
    return prewalk(rmlines, flatten(Data_module))
end

function TData_cuda()
    TData_module = :(
        baremodule $MODULENAME_TDATA
            import Base, CUDA, ParallelStencil.ParallelKernel.CellArrays, ParallelStencil.ParallelKernel.StaticArrays
            # TODO: the constructors defined by CellArrays.@define_CuCellArray lead to pre-compilation issues due to a bug in Julia. We therefore only create the type alias here for now.
            const CuCellArray{T,N,B,T_elem} = CellArrays.CellArray{T,N,B,CUDA.CuArray{T_elem,CellArrays._N}}
            # CellArrays.@define_CuCellArray
            # export CuCellArray
            const Array{T, N}               = CUDA.CuArray{T, N}
            const Cell{T, S}                = Union{StaticArrays.SArray{S, T}, StaticArrays.FieldArray{S, T}}
            const CellArray{T_elem, N, B}   = CuCellArray{<:Cell{T_elem},N,B,T_elem}
            $(TData_xpu_exprs())
            $(TData_Device_cuda())
            $(TData_Fields())
        end
    )
    return prewalk(rmlines, flatten(TData_module))
end

function Data_Device_cuda(numbertype::DataType, indextype::DataType)
    Device_module = if (numbertype == NUMBERTYPE_NONE)
        :(baremodule $MODULENAME_DEVICE
            import Base, CUDA, ParallelStencil.ParallelKernel.CellArrays, ParallelStencil.ParallelKernel.StaticArrays
            const Index                     = $indextype
            const Array{T, N}               = CUDA.CuDeviceArray{T, N}
            const Cell{T, S}                = Union{StaticArrays.SArray{S, T}, StaticArrays.FieldArray{S, T}}
            const CellArray{T_elem, N, B}   = CellArrays.CellArray{<:Cell{T_elem},N,B,<:CUDA.CuDeviceArray{T_elem,CellArrays._N}}
            $(Data_xpu_exprs(numbertype))
        end)
    else
        :(baremodule $MODULENAME_DEVICE
            import Base, CUDA, ParallelStencil.ParallelKernel.CellArrays, ParallelStencil.ParallelKernel.StaticArrays
            const Index                     = $indextype
            const Array{N}                  = CUDA.CuDeviceArray{$numbertype, N}
            const Cell{S}                   = Union{StaticArrays.SArray{S, $numbertype}, StaticArrays.FieldArray{S, $numbertype}}
            const CellArray{N, B}           = CellArrays.CellArray{<:Cell,N,B,<:CUDA.CuDeviceArray{$numbertype,CellArrays._N}}
            $(Data_xpu_exprs(numbertype))
        end)
    end
    return Device_module
end

function TData_Device_cuda()
    :(baremodule $MODULENAME_DEVICE
        import Base, CUDA, ParallelStencil.ParallelKernel.CellArrays, ParallelStencil.ParallelKernel.StaticArrays
        const Array{T, N}                   = CUDA.CuDeviceArray{T, N}
        const Cell{T, S}                    = Union{StaticArrays.SArray{S, T}, StaticArrays.FieldArray{S, T}}
        const CellArray{T_elem, N, B}       = CellArrays.CellArray{<:Cell{T_elem},N,B,<:CUDA.CuDeviceArray{T_elem,CellArrays._N}}
        $(TData_xpu_exprs())
    end)
end


# AMDGPU

function Data_amdgpu(numbertype::DataType, indextype::DataType)
    Data_module = if (numbertype == NUMBERTYPE_NONE)
        :(baremodule $MODULENAME_DATA
            import Base, AMDGPU, ParallelStencil.ParallelKernel.CellArrays, ParallelStencil.ParallelKernel.StaticArrays
            # TODO: the constructors defined by CellArrays.@define_ROCCellArray lead to pre-compilation issues due to a bug in Julia. We therefore only create the type alias here for now.
            const ROCCellArray{T,N,B,T_elem} = CellArrays.CellArray{T,N,B,AMDGPU.ROCArray{T_elem,CellArrays._N}}
            # CellArrays.@define_ROCCellArray
            # export ROCCellArray
            const Index                      = $indextype
            const Array{T, N}                = AMDGPU.ROCArray{T, N}
            const Cell{T, S}                 = Union{StaticArrays.SArray{S, T}, StaticArrays.FieldArray{S, T}}
            const CellArray{T_elem, N, B}    = ROCCellArray{<:Cell{T_elem},N,B,T_elem}
            $(Data_xpu_exprs(numbertype))
            $(Data_Device_amdgpu(numbertype, indextype))
            $(Data_Fields(numbertype, indextype))
        end)
    else
        :(baremodule $MODULENAME_DATA
            import Base, AMDGPU, ParallelStencil.ParallelKernel.CellArrays, ParallelStencil.ParallelKernel.StaticArrays
            # TODO: the constructors defined by CellArrays.@define_ROCCellArray lead to pre-compilation issues due to a bug in Julia. We therefore only create the type alias here for now.
            const ROCCellArray{T,N,B,T_elem} = CellArrays.CellArray{T,N,B,AMDGPU.ROCArray{T_elem,CellArrays._N}}
            # CellArrays.@define_ROCCellArray
            # export ROCCellArray
            const Index                      = $indextype
            const Number                     = $numbertype
            const Array{N}                   = AMDGPU.ROCArray{$numbertype, N}
            const Cell{S}                    = Union{StaticArrays.SArray{S, $numbertype}, StaticArrays.FieldArray{S, $numbertype}}
            const CellArray{N, B}            = ROCCellArray{<:Cell,N,B,$numbertype}
            $(Data_xpu_exprs(numbertype))
            $(Data_Device_amdgpu(numbertype, indextype))
            $(Data_Fields(numbertype, indextype))
        end)
    end
    return prewalk(rmlines, flatten(Data_module))
end

function TData_amdgpu()
    TData_module = :(
        baremodule $MODULENAME_TDATA
            import Base, AMDGPU, ParallelStencil.ParallelKernel.CellArrays, ParallelStencil.ParallelKernel.StaticArrays
            # TODO: the constructors defined by CellArrays.@define_ROCCellArray lead to pre-compilation issues due to a bug in Julia. We therefore only create the type alias here for now.
            const ROCCellArray{T,N,B,T_elem} = CellArrays.CellArray{T,N,B,AMDGPU.ROCArray{T_elem,CellArrays._N}}
            # CellArrays.@define_ROCCellArray
            # export ROCCellArray
            const Array{T, N}                = AMDGPU.ROCArray{T, N}
            const Cell{T, S}                 = Union{StaticArrays.SArray{S, T}, StaticArrays.FieldArray{S, T}}
            const CellArray{T_elem, N, B}    = ROCCellArray{<:Cell{T_elem},N,B,T_elem}
            $(TData_xpu_exprs())
            $(TData_Device_amdgpu())
            $(TData_Fields())
        end
        )
    return prewalk(rmlines, flatten(TData_module))
end

function Data_Device_amdgpu(numbertype::DataType, indextype::DataType)
    Device_module = if (numbertype == NUMBERTYPE_NONE)
        :(baremodule $MODULENAME_DEVICE
            import Base, AMDGPU, ParallelStencil.ParallelKernel.CellArrays, ParallelStencil.ParallelKernel.StaticArrays
            const Index                      = $indextype
            const Array{T, N}                = AMDGPU.ROCDeviceArray{T, N}
            const Cell{T, S}                 = Union{StaticArrays.SArray{S, T}, StaticArrays.FieldArray{S, T}}
            const CellArray{T_elem, N, B}    = CellArrays.CellArray{<:Cell{T_elem},N,B,<:AMDGPU.ROCDeviceArray{T_elem,CellArrays._N}}
            $(Data_xpu_exprs(numbertype))
        end)
    else
        :(baremodule $MODULENAME_DEVICE
            import Base, AMDGPU, ParallelStencil.ParallelKernel.CellArrays, ParallelStencil.ParallelKernel.StaticArrays
            const Index                      = $indextype
            const Array{N}                   = AMDGPU.ROCDeviceArray{$numbertype, N}
            const Cell{S}                    = Union{StaticArrays.SArray{S, $numbertype}, StaticArrays.FieldArray{S, $numbertype}}
            const CellArray{N, B}            = CellArrays.CellArray{<:Cell,N,B,<:AMDGPU.ROCDeviceArray{$numbertype,CellArrays._N}}
            $(Data_xpu_exprs(numbertype))
        end)
    end
    return Device_module
end

function TData_Device_amdgpu()
    :(baremodule $MODULENAME_DEVICE
        import Base, AMDGPU, ParallelStencil.ParallelKernel.CellArrays, ParallelStencil.ParallelKernel.StaticArrays
        const Array{T, N}                    = AMDGPU.ROCDeviceArray{T, N}
        const Cell{T, S}                     = Union{StaticArrays.SArray{S, T}, StaticArrays.FieldArray{S, T}}
        const CellArray{T_elem, N, B}        = CellArrays.CellArray{<:Cell{T_elem},N,B,<:AMDGPU.ROCDeviceArray{T_elem,CellArrays._N}}
        $(TData_xpu_exprs())
    end)
end

# Metal

function Data_metal(numbertype::DataType, indextype::DataType)
    Data_module = if (numbertype == NUMBERTYPE_NONE)
        :(baremodule $MODULENAME_DATA # NOTE: there cannot be any newline before 'module Data' or it will create a begin end block and the module creation will fail.
            import Base, Metal, ParallelStencil.ParallelKernel.CellArrays, ParallelStencil.ParallelKernel.StaticArrays
            const MtlCellArray{T,N,B,T_elem} = CellArrays.CellArray{T,N,B,Metal.MtlArray{T_elem,CellArrays._N}}
            const Index                     = $indextype
            const Array{T, N}               = Metal.MtlArray{T, N}
            const Cell{T, S}                = Union{StaticArrays.SArray{S, T}, StaticArrays.FieldArray{S, T}}
            const CellArray{T_elem, N, B}   = MtlCellArray{<:Cell{T_elem},N,B,T_elem}
            $(Data_xpu_exprs(numbertype)) 
            $(Data_Device_metal(numbertype, indextype))
            $(Data_Fields(numbertype, indextype))
        end)
    else
        :(baremodule $MODULENAME_DATA
            import Base, Metal, ParallelStencil.ParallelKernel.CellArrays, ParallelStencil.ParallelKernel.StaticArrays
            const MtlCellArray{T,N,B,T_elem} = CellArrays.CellArray{T,N,B,Metal.MtlArray{T_elem,CellArrays._N}}
            const Index                     = $indextype
            const Number                    = $numbertype
            const Array{N}                  = Metal.MtlArray{$numbertype, N}
            const Cell{S}                   = Union{StaticArrays.SArray{S, $numbertype}, StaticArrays.FieldArray{S, $numbertype}}
            const CellArray{N, B}           = MtlCellArray{<:Cell,N,B,$numbertype}
            $(Data_xpu_exprs(numbertype))
            $(Data_Device_metal(numbertype, indextype))
            $(Data_Fields(numbertype, indextype))
        end)
    end
    return prewalk(rmlines, flatten(Data_module))
end

function TData_metal()
    TData_module = :(
        baremodule $MODULENAME_TDATA
            import Base, Metal, ParallelStencil.ParallelKernel.CellArrays, ParallelStencil.ParallelKernel.StaticArrays
            const MtlCellArray{T,N,B,T_elem} = CellArrays.CellArray{T,N,B,Metal.MtlArray{T_elem,CellArrays._N}}
            const Array{T, N}               = Metal.MtlArray{T, N}
            const Cell{T, S}                = Union{StaticArrays.SArray{S, T}, StaticArrays.FieldArray{S, T}}
            const CellArray{T_elem, N, B}   = MtlCellArray{<:Cell{T_elem},N,B,T_elem}
            $(TData_xpu_exprs())
            $(TData_Device_metal())
            $(TData_Fields())
        end
    )
    return prewalk(rmlines, flatten(TData_module))
end

function Data_Device_metal(numbertype::DataType, indextype::DataType)
    Device_module = if (numbertype == NUMBERTYPE_NONE)
        :(baremodule $MODULENAME_DEVICE
            import Base, Metal, ParallelStencil.ParallelKernel.CellArrays, ParallelStencil.ParallelKernel.StaticArrays
            const Index                     = $indextype
            const Array{T, N}               = Metal.MtlDeviceArray{T, N}
            const Cell{T, S}                = Union{StaticArrays.SArray{S, T}, StaticArrays.FieldArray{S, T}}
            const CellArray{T_elem, N, B}   = CellArrays.CellArray{<:Cell{T_elem},N,B,<:Metal.MtlDeviceArray{T_elem,CellArrays._N}}
            $(Data_xpu_exprs(numbertype))
        end)
    else
        :(baremodule $MODULENAME_DEVICE
            import Base, Metal, ParallelStencil.ParallelKernel.CellArrays, ParallelStencil.ParallelKernel.StaticArrays
            const Index                     = $indextype
            const Array{N}                  = Metal.MtlDeviceArray{$numbertype, N}
            const Cell{S}                   = Union{StaticArrays.SArray{S, $numbertype}, StaticArrays.FieldArray{S, $numbertype}}
            const CellArray{N, B}           = CellArrays.CellArray{<:Cell,N,B,<:Metal.MtlDeviceArray{$numbertype,CellArrays._N}}
            $(Data_xpu_exprs(numbertype))
        end)
    end
    return Device_module
end

function TData_Device_metal()
    :(baremodule $MODULENAME_DEVICE
        import Base, Metal, ParallelStencil.ParallelKernel.CellArrays, ParallelStencil.ParallelKernel.StaticArrays
        const Array{T, N}                   = Metal.MtlDeviceArray{T, N}
        const Cell{T, S}                    = Union{StaticArrays.SArray{S, T}, StaticArrays.FieldArray{S, T}}
        const CellArray{T_elem, N, B}       = CellArrays.CellArray{<:Cell{T_elem},N,B,<:Metal.MtlDeviceArray{T_elem,CellArrays._N}}
        $(TData_xpu_exprs())
    end)
end

# CPU

function Data_cpu(numbertype::DataType, indextype::DataType)
    Data_module = if (numbertype == NUMBERTYPE_NONE)
        :(baremodule $MODULENAME_DATA
            import Base, ParallelStencil.ParallelKernel.CellArrays, ParallelStencil.ParallelKernel.StaticArrays
            const Index                      = $indextype
            const Array{T, N}                = Base.Array{T, N}
            const Cell{T, S}                 = Union{StaticArrays.SArray{S, T}, StaticArrays.FieldArray{S, T}}
            const CellArray{T_elem, N, B}    = CellArrays.CPUCellArray{<:Cell{T_elem},N,B,T_elem}
            $(Data_xpu_exprs(numbertype))
            $(Data_Device_cpu(numbertype, indextype))
            $(Data_Fields(numbertype, indextype))
        end)
    else
        :(baremodule $MODULENAME_DATA
            import Base, ParallelStencil.ParallelKernel.CellArrays, ParallelStencil.ParallelKernel.StaticArrays
            const Index                      = $indextype
            const Number                     = $numbertype
            const Array{N}                   = Base.Array{$numbertype, N}
            const Cell{S}                    = Union{StaticArrays.SArray{S, $numbertype}, StaticArrays.FieldArray{S, $numbertype}}
            const CellArray{N, B}            = CellArrays.CPUCellArray{<:Cell,N,B,$numbertype}
            $(Data_xpu_exprs(numbertype))
            $(Data_Device_cpu(numbertype, indextype))
            $(Data_Fields(numbertype, indextype))
        end)
    end
    return prewalk(rmlines, flatten(Data_module))
end

function TData_cpu()
    TData_module = :(
        baremodule $MODULENAME_TDATA
            import Base, ParallelStencil.ParallelKernel.CellArrays, ParallelStencil.ParallelKernel.StaticArrays
            const Array{T, N}                = Base.Array{T, N}
            const Cell{T, S}                 = Union{StaticArrays.SArray{S, T}, StaticArrays.FieldArray{S, T}}
            const CellArray{T_elem, N, B}    = CellArrays.CPUCellArray{<:Cell{T_elem},N,B,T_elem}
            $(TData_xpu_exprs())
            $(TData_Device_cpu())
            $(TData_Fields())
        end
    )
    return prewalk(rmlines, flatten(TData_module))
end

function Data_Device_cpu(numbertype::DataType, indextype::DataType)
    Device_module = if (numbertype == NUMBERTYPE_NONE)
        :(baremodule $MODULENAME_DEVICE
            import Base, ParallelStencil.ParallelKernel.CellArrays, ParallelStencil.ParallelKernel.StaticArrays
            const Index                      = $indextype
            const Array{T, N}                = Base.Array{T, N}
            const Cell{T, S}                 = Union{StaticArrays.SArray{S, T}, StaticArrays.FieldArray{S, T}}
            const CellArray{T_elem, N, B}    = CellArrays.CPUCellArray{<:Cell{T_elem},N,B,T_elem}
            $(Data_xpu_exprs(numbertype))
        end)
    else
        :(baremodule $MODULENAME_DEVICE
            import Base, ParallelStencil.ParallelKernel.CellArrays, ParallelStencil.ParallelKernel.StaticArrays
            const Index                      = $indextype
            const Array{N}                   = Base.Array{$numbertype, N}
            const Cell{S}                    = Union{StaticArrays.SArray{S, $numbertype}, StaticArrays.FieldArray{S, $numbertype}}
            const CellArray{N, B}            = CellArrays.CPUCellArray{<:Cell,N,B,$numbertype}
            $(Data_xpu_exprs(numbertype))
        end)
    end
    return Device_module
end

function TData_Device_cpu()
    :(baremodule $MODULENAME_DEVICE
        import Base, ParallelStencil.ParallelKernel.CellArrays, ParallelStencil.ParallelKernel.StaticArrays
        const Array{T, N}                    = Base.Array{T, N}
        const Cell{T, S}                     = Union{StaticArrays.SArray{S, T}, StaticArrays.FieldArray{S, T}}
        const CellArray{T_elem, N, B}        = CellArrays.CPUCellArray{<:Cell{T_elem},N,B,T_elem}
        $(TData_xpu_exprs())
    end)
end


# xPU

function Data_xpu_exprs(numbertype::DataType)
    if (numbertype == NUMBERTYPE_NONE) T_xpu_exprs()
    else                               xpu_exprs()
    end
end

TData_xpu_exprs() = T_xpu_exprs()

function T_xpu_exprs()
    quote
        const NumberTuple{N_tuple, T}                           = NTuple{N_tuple, T}
        const ArrayTuple{N_tuple, T, N}                         = NTuple{N_tuple, Array{T, N}}
        const CellTuple{N_tuple, T, S}                          = NTuple{N_tuple, Cell{T, S}}
        const CellArrayTuple{N_tuple, T_elem, N, B}             = NTuple{N_tuple, CellArray{T_elem, N, B}}

        const NamedNumberTuple{N_tuple, T, names}               = NamedTuple{names, <:NumberTuple{N_tuple, T}}
        const NamedArrayTuple{N_tuple, T, N, names}             = NamedTuple{names, <:ArrayTuple{N_tuple, T, N}}
        const NamedCellTuple{N_tuple, T, S, names}              = NamedTuple{names, <:CellTuple{N_tuple, T, S}}
        const NamedCellArrayTuple{N_tuple, T_elem, N, B, names} = NamedTuple{names, <:CellArrayTuple{N_tuple, T_elem, N, B}}

        const NumberCollection{N_tuple, T}                       = Union{NumberTuple{N_tuple, T}, NamedNumberTuple{N_tuple, T}}
        const ArrayCollection{N_tuple, T, N}                     = Union{ArrayTuple{N_tuple, T, N}, NamedArrayTuple{N_tuple, T, N}}
        const CellCollection{N_tuple, T, S}                      = Union{CellTuple{N_tuple, T, S}, NamedCellTuple{N_tuple, T, S}}
        const CellArrayCollection{N_tuple, T_elem, N, B}         = Union{CellArrayTuple{N_tuple, T_elem, N, B}, NamedCellArrayTuple{N_tuple, T_elem, N, B}}        

        # TODO: the following constructors lead to pre-compilation issues due to a bug in Julia. They are therefore commented out for now.
        # NamedNumberTuple{}(T, t::NamedTuple)                     = Base.map(T, t)
        # NamedArrayTuple{}(T, t::NamedTuple)                      = Base.map(Data.Array{T}, t)
        # NamedCellTuple{}(T, t::NamedTuple)                       = Base.map(Data.Cell{T}, t)
        # NamedCellArrayTuple{}(T, t::NamedTuple)                  = Base.map(Data.CellArray{T}, t)
    end
end

function xpu_exprs()
    quote
        const IndexTuple{N_tuple}                                = NTuple{N_tuple, Index}
        const NumberTuple{N_tuple}                               = NTuple{N_tuple, Number}
        const ArrayTuple{N_tuple, N}                             = NTuple{N_tuple, Array{N}}
        const CellTuple{N_tuple, S}                              = NTuple{N_tuple, Cell{S}}
        const CellArrayTuple{N_tuple, N, B}                      = NTuple{N_tuple, CellArray{N, B}}

        const NamedIndexTuple{N_tuple, names}                    = NamedTuple{names, <:IndexTuple{N_tuple}}
        const NamedNumberTuple{N_tuple, names}                   = NamedTuple{names, <:NumberTuple{N_tuple}}
        const NamedArrayTuple{N_tuple, N, names}                 = NamedTuple{names, <:ArrayTuple{N_tuple, N}}
        const NamedCellTuple{N_tuple, S, names}                  = NamedTuple{names, <:CellTuple{N_tuple, S}}
        const NamedCellArrayTuple{N_tuple, N, B, names}          = NamedTuple{names, <:CellArrayTuple{N_tuple, N, B}}

        const IndexCollection{N_tuple}                           = Union{IndexTuple{N_tuple}, NamedIndexTuple{N_tuple}}
        const NumberCollection{N_tuple}                          = Union{NumberTuple{N_tuple}, NamedNumberTuple{N_tuple}}
        const ArrayCollection{N_tuple, N}                        = Union{ArrayTuple{N_tuple, N}, NamedArrayTuple{N_tuple, N}}
        const CellCollection{N_tuple, S}                         = Union{CellTuple{N_tuple, S}, NamedCellTuple{N_tuple, S}}
        const CellArrayCollection{N_tuple, N, B}                 = Union{CellArrayTuple{N_tuple, N, B}, NamedCellArrayTuple{N_tuple, N, B}}
        
        # TODO: the following constructors lead to pre-compilation issues due to a bug in Julia. They are therefore commented out for now.
        # NamedIndexTuple{}(t::NamedTuple)                         = Base.map(Data.Index, t)
        # NamedNumberTuple{}(t::NamedTuple)                        = Base.map(Data.Number, t)
        # NamedArrayTuple{}(t::NamedTuple)                         = Base.map(Data.Array, t)
        # NamedCellTuple{}(t::NamedTuple)                          = Base.map(Data.Cell, t)
        # NamedCellArrayTuple{}(t::NamedTuple)                     = Base.map(Data.CellArray, t)
    end
end


## (DATA SUBMODULE FIELDS - xPU)  # NOTE: custom data types could be implemented for each alias.

function Data_Fields(numbertype::DataType, indextype::DataType)
    Fields_module = if (numbertype == NUMBERTYPE_NONE)
        :(baremodule $MODULENAME_FIELDS
            import ..$MODULENAME_DATA                          # NOTE: this requires Julia >=1.10
            import ..$MODULENAME_DATA: Array, NamedArrayTuple
            $(generic_Fields_exprs())
            $(T_Fields_exprs())
            $(Data_Fields_Device(numbertype, indextype))
        end)
    else
        :(baremodule $MODULENAME_FIELDS
            import ..$MODULENAME_DATA
            import ..$MODULENAME_DATA: Array, NamedArrayTuple
            $(generic_Fields_exprs())
            $(Fields_exprs())
            $(Data_Fields_Device(numbertype, indextype))
        end)
    end
    return Fields_module
end

function TData_Fields()
    :(baremodule $MODULENAME_FIELDS
        import ..$MODULENAME_TDATA
        import ..$MODULENAME_TDATA: Array, NamedArrayTuple
        $(generic_Fields_exprs())
        $(T_Fields_exprs())
        $(TData_Fields_Device())
    end)
end

function Data_Fields_Device(numbertype::DataType, indextype::DataType)
    Device_module = if (numbertype == NUMBERTYPE_NONE)
        :(baremodule $MODULENAME_DEVICE
            import ..$MODULENAME_DATA.$MODULENAME_DEVICE: Array, NamedArrayTuple
            $(generic_Fields_exprs())
            $(T_Fields_exprs())
        end)
    else
        :(baremodule $MODULENAME_DEVICE
            import ..$MODULENAME_DATA.$MODULENAME_DEVICE: Array, NamedArrayTuple
            $(generic_Fields_exprs())
            $(Fields_exprs())
        end)
    end
    return Device_module
end

function TData_Fields_Device()
    :(baremodule $MODULENAME_DEVICE
        import ..$MODULENAME_TDATA.$MODULENAME_DEVICE: Array, NamedArrayTuple
        $(generic_Fields_exprs())
        $(T_Fields_exprs())
    end)
end

function T_Fields_exprs()
    quote
        export VectorField, BVectorField, TensorField
        const VectorField{T, N, names}  = NamedArrayTuple{N, T, N, names}
        const BVectorField{T, N, names} = NamedArrayTuple{N, T, N, names}
        const TensorField{T, N, names}  = NamedArrayTuple{N, T, N, names}
    end
end

function Fields_exprs()
    quote
        export VectorField, BVectorField, TensorField
        const VectorField{N, names}     = NamedArrayTuple{N, N, names}
        const BVectorField{N, names}    = NamedArrayTuple{N, N, names}
        const TensorField{N, names}     = NamedArrayTuple{N, N, names}
    end
end

function generic_Fields_exprs()
    quote
        export Field, XField, YField, ZField, BXField, BYField, BZField, XXField, YYField, ZZField, XYField, XZField, YZField
        const Field                     = Array
        const XField                    = Array
        const YField                    = Array
        const ZField                    = Array
        const BXField                   = Array
        const BYField                   = Array
        const BZField                   = Array
        const XXField                   = Array
        const YYField                   = Array
        const ZZField                   = Array
        const XYField                   = Array
        const XZField                   = Array
        const YZField                   = Array
    end
end
