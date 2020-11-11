"""
Weak scaling test for assembly routines in FEniCS/DOLFINX (<https://www.fenicsproject.org/>).

NOTE: This test is based on [the unit test] from FEniCS/DOLFINX.

[the unit test]: https://github.com/FEniCS/dolfinx/blob/939a235be7dc8808b1f09d0432e433b089e67f1e/python/test/unit/fem/test_nonlinear_assembler.py
"""

# --------------------------------------------------------------------------------------------------
# Monkeypatch for running tests on ULHPC cluster Iris (https://hpc.uni.lu/systems/iris/)

# NOTE:
#   Some 3rd party libraries (e.g. gmsh, tqdm) call methods from the built-in `platform` module
#   which can lead to dangerous forking via subprocess, see https://github.com/tqdm/tqdm/pull/1046.
#
#   See also:
#     - https://github.com/tqdm/tqdm/issues/691
#     - https://github.com/h5py/h5py/issues/1079

import socket

if socket.gethostname().startswith("iris"):
    import os
    import platform
    from collections import namedtuple

    uname_result = namedtuple("uname_result", "system node release version machine processor")
    system, node, release, version, machine = os.uname()
    assert platform._uname_cache is None
    platform._uname_cache = uname_result(system, node, release, version, machine, machine)
# --------------------------------------------------------------------------------------------------

from mpi4py import MPI
from petsc4py import PETSc

import numpy
import ufl
import dolfinx


def nest_matrix_norm(A):
    """Return norm of a MatNest matrix"""
    import math

    assert A.getType() == "nest"
    norm = 0.0
    nrows, ncols = A.getNestSize()
    for row in range(nrows):
        for col in range(ncols):
            A_sub = A.getNestSubMatrix(row, col)
            if A_sub:
                _norm = A_sub.norm()
                norm += _norm * _norm
    return math.sqrt(norm)


def target_mesh_size(comm_size, num_coredofs=30000):
    """Return mesh resolution on the edge of the unit cube so that the number of DOFs per core is
    sufficiently close to the required value passed in as `num_coredofs`."""
    # Get analytic formula for total number of DOFs (for both tetrahedron and hexahedron)
    num_dofs = lambda N: 3 * (2 * N + 1) ** 3 + 1 * (N + 1) ** 3  # noqa: E731
    polycoeffs = [25, 39, 21, 4]
    assert numpy.polyval(polycoeffs, 10) == num_dofs(10)

    # Estimate edge resolution
    candidates = numpy.roots(polycoeffs[:-1] + [polycoeffs[-1] - num_coredofs * comm_size])
    N = [int(numpy.round(N.real)) for N in candidates if numpy.isreal(N)]
    assert len(N) == 1

    return N[0]


def monolithic_assembly(clock, reps, mesh, lifting):
    P2_el = ufl.VectorElement("Lagrange", mesh.ufl_cell(), 2)
    P1_el = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    TH = P2_el * P1_el
    W = dolfinx.FunctionSpace(mesh, TH)
    num_dofs = W.dim

    U = dolfinx.Function(W)
    u, p = ufl.split(U)
    v, q = ufl.TestFunctions(W)

    g = ufl.as_vector([0.0, 0.0, -1.0])
    F = (
        ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(p, ufl.div(v)) * ufl.dx
        + ufl.inner(ufl.div(u), q) * ufl.dx
        - ufl.inner(g, v) * ufl.dx
    )
    J = ufl.derivative(F, U, ufl.TrialFunction(W))
    bcs = []

    # Get jitted forms for better performance
    F = dolfinx.fem.assembler._create_cpp_form(F)
    J = dolfinx.fem.assembler._create_cpp_form(J)

    b = dolfinx.fem.create_vector(F)
    A = dolfinx.fem.create_matrix(J)
    for i in range(reps):
        A.zeroEntries()
        with b.localForm() as b_local:
            b_local.set(0.0)

        with dolfinx.common.Timer("ZZZ Mat Monolithic") as tmr:
            dolfinx.fem.assemble_matrix(A, J, bcs)
            A.assemble()
            clock["mat"] += tmr.elapsed()[0]

        with dolfinx.common.Timer("ZZZ Vec Monolithic") as tmr:
            dolfinx.fem.assemble_vector(b, F)
            if lifting:
                dolfinx.fem.apply_lifting(b, [J], bcs=[bcs], x0=[U.vector], scale=-1.0)
            b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            if lifting:
                dolfinx.fem.set_bc(b, bcs, x0=U.vector, scale=-1.0)
                b.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            clock["vec"] += tmr.elapsed()[0]

    return num_dofs, A, b


def block_assembly(clock, reps, mesh, lifting, nest=False):
    P2 = dolfinx.function.VectorFunctionSpace(mesh, ("Lagrange", 2))
    P1 = dolfinx.function.FunctionSpace(mesh, ("Lagrange", 1))
    num_dofs = P2.dim + P1.dim

    g = ufl.as_vector([0.0, 0.0, -1.0])
    u, p = dolfinx.Function(P2), dolfinx.Function(P1)
    du, dp = ufl.TrialFunction(P2), ufl.TrialFunction(P1)
    v, q = ufl.TestFunction(P2), ufl.TestFunction(P1)

    F = [
        ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(p, ufl.div(v)) * ufl.dx
        - ufl.inner(g, v) * ufl.dx,
        ufl.inner(ufl.div(u), q) * ufl.dx,
    ]
    J = [
        [ufl.derivative(F[0], u, du), ufl.derivative(F[0], p, dp)],
        [ufl.derivative(F[1], u, du), ufl.derivative(F[1], p, dp)],
    ]
    bcs = []

    # Get jitted forms for better performance
    F = dolfinx.fem.assembler._create_cpp_form(F)
    J = dolfinx.fem.assembler._create_cpp_form(J)

    if nest:
        x0 = dolfinx.fem.create_vector_nest(F)
        b = dolfinx.fem.create_vector_nest(F)
        A = dolfinx.fem.create_matrix_nest(J)
        for i in range(reps):
            A.zeroEntries()
            for b_sub in b.getNestSubVecs():
                with b_sub.localForm() as b_local:
                    b_local.set(0.0)

            with dolfinx.common.Timer("ZZZ Mat Nest") as tmr:
                dolfinx.fem.assemble_matrix_nest(A, J, bcs)
                A.assemble()
                clock["mat"] += tmr.elapsed()[0]

            with dolfinx.common.Timer("ZZZ Vec Nest") as tmr:
                dolfinx.fem.assemble_vector_nest(b, F)
                if lifting:
                    dolfinx.fem.apply_lifting_nest(b, J, bcs, x0, scale=-1.0)
                for b_sub in b.getNestSubVecs():
                    b_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
                if lifting:
                    bcs0 = dolfinx.cpp.fem.bcs_rows(dolfinx.fem.assemble._create_cpp_form(F), bcs)
                    dolfinx.fem.set_bc_nest(b, bcs0, x0, scale=-1.0)
                clock["vec"] += tmr.elapsed()[0]
    else:
        from dolfinx.fem.assemble import _create_cpp_form

        x0 = dolfinx.fem.create_vector_block(F)
        b = dolfinx.fem.create_vector_block(F)
        A = dolfinx.fem.create_matrix_block(J)
        for i in range(reps):
            A.zeroEntries()
            with b.localForm() as b_local:
                b_local.set(0.0)

            with dolfinx.common.Timer("ZZZ Mat Block") as tmr:
                dolfinx.fem.assemble_matrix_block(A, J, bcs)
                A.assemble()
                clock["mat"] += tmr.elapsed()[0]

            # # NOTE: Ghosts are updated inside assemble_vector_block
            # with dolfinx.common.Timer("ZZZ Vec Block") as tmr:
            #     dolfinx.fem.assemble_vector_block(b, F, J, bcs, x0=x0, scale=-1.0)
            #     clock["vec"] += tmr.elapsed()[0]

            # NOTE: The following code does exactly the same thing as
            #       dolfinx.fem.assemble_vector_block(b, F, J, bcs, x0=x0, scale=-1.0)
            a, L = J, F
            scale = -1.0
            with dolfinx.common.Timer("ZZZ Vec Block") as tmr:
                maps = [form.function_spaces[0].dofmap.index_map for form in _create_cpp_form(L)]
                if x0 is not None:
                    x0_local = dolfinx.cpp.la.get_local_vectors(x0, maps)
                    x0_sub = x0_local
                else:
                    x0_local = []
                    x0_sub = [None] * len(maps)

                bcs1 = dolfinx.cpp.fem.bcs_cols(_create_cpp_form(a), bcs) if lifting else len(L) * [None]
                b_local = dolfinx.cpp.la.get_local_vectors(b, maps)
                for b_sub, L_sub, a_sub, bc in zip(b_local, L, a, bcs1):
                    dolfinx.cpp.fem.assemble_vector(b_sub, _create_cpp_form(L_sub))
                    if lifting:
                        dolfinx.cpp.fem.apply_lifting(b_sub, _create_cpp_form(a_sub), bc, x0_local, scale)

                dolfinx.cpp.la.scatter_local_vectors(b, b_local, maps)
                b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

                if lifting:
                    bcs0 = dolfinx.cpp.fem.bcs_rows(_create_cpp_form(L), bcs)
                    offset = 0
                    b_array = b.getArray(readonly=False)
                    for submap, bc, _x0 in zip(maps, bcs0, x0_sub):
                        size = submap.size_local * submap.block_size
                        dolfinx.cpp.fem.set_bc(b_array[offset:offset + size], bc, _x0, scale)
                        offset += size
                clock["vec"] += tmr.elapsed()[0]

    return num_dofs, A, b


def test_assembler(
    atype="mono",
    reps=1,
    num_coredofs=30000,
    lifting=True,
    overwrite=True,
    results_file="results_assembly_routines.csv",
):
    """Test chosen assembly routine on Taylor-Hood elements."""
    N = target_mesh_size(MPI.COMM_WORLD.size, num_coredofs)
    PETSc.Sys.Print(f"Generating unit cube mesh with edge resolution N = {N}")
    mesh = dolfinx.generation.UnitCubeMesh(MPI.COMM_WORLD, N, N, N)
    comm = mesh.mpi_comm()

    clock = {
        "mat": 0.0,
        "vec": 0.0,
    }
    if atype == "mono":
        num_dofs, A, b = monolithic_assembly(clock, reps, mesh, lifting)
    elif atype == "block":
        num_dofs, A, b = block_assembly(clock, reps, mesh, lifting)
    elif atype == "nest":
        num_dofs, A, b = block_assembly(clock, reps, mesh, lifting, nest=True)

    # Evaluate norms
    A_norm = nest_matrix_norm(A) if atype == "nest" else A.norm()
    b_norm = b.norm()

    # Check number of DOFs per core
    if atype != "nest":
        core_dofs = b.local_size
        with b.localForm() as b_local:
            core_gdofs = b_local.local_size
    else:
        core_dofs = 0
        core_gdofs = 0
        for b_sub in b.getNestSubVecs():
            core_dofs += b_sub.local_size
            with b_sub.localForm() as b_local:
                core_gdofs += b_local.local_size

    # List timings
    dolfinx.common.list_timings(comm, [dolfinx.common.TimingType.wall])
    PETSc.Sys.Print(f"Matrix norm: {A_norm}")
    PETSc.Sys.Print(f"Vector norm: {b_norm}")
    PETSc.Sys.Print(f"Average number of DOFs per process: {num_dofs / comm.size}")

    # Postprocess
    has_pandas = True
    try:
        import pandas
    except ModuleNotFoundError:
        has_pandas = False

    if has_pandas:
        import os

        results = {
            "assembler": atype,
            "mesh_resolution": N,
            "num_procs": comm.size,
            "num_reps": reps,
            "num_dofs": num_dofs,
            "avg_core_dofs": comm.allreduce(core_dofs, op=MPI.SUM) / comm.size,
            "min_core_dofs": comm.allreduce(core_dofs, op=MPI.MIN),
            "max_core_dofs": comm.allreduce(core_dofs, op=MPI.MAX),
            "min_core_ghosts": comm.allreduce(core_gdofs - core_dofs, op=MPI.MIN),
            "max_core_ghosts": comm.allreduce(core_gdofs - core_dofs, op=MPI.MAX),
            "mat_norm": A_norm,
            "vec_norm": b_norm,
        }
        for key in clock.keys():
            results[f"t_{key}"] = comm.allreduce(clock[key], op=MPI.SUM) / comm.size
            results[f"t_{key}_dof"] = comm.allreduce(clock[key] / core_dofs, op=MPI.SUM) / comm.size

        if comm.rank == 0:
            data = pandas.DataFrame(results, index=[0])
            if overwrite:
                mode = "w"
                header = True
            else:
                mode = "a"
                header = not os.path.exists(results_file)

            data.to_csv(results_file, index=False, mode=mode, header=header)


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Weak scaling test for assembly routines in FEniCS/DOLFINX",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-t", type=str, default="mono", dest="atype", choices=["mono", "block", "nest"], help="type of assembly routine")
    parser.add_argument("-r", type=int, default=1, help="number of assembly repetitions")
    parser.add_argument("--dofs", type=int, default=30000, help="number of DOFs per core")
    parser.add_argument("--no-lifting", action="store_false", dest="lifting", help="do not apply any BC-related lifting")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing results file")
    parser.add_argument("--results", type=str, metavar="FILENAME", default="results_assembly_routines.csv", help="CSV file to store the results, requires pandas")
    args = parser.parse_args(sys.argv[1:])

    sys.exit(test_assembler(args.atype, args.r, args.dofs, args.lifting, args.overwrite, args.results))
