import os
import sys

# Add local path
sys.path.append(os.getcwd())

try:
    from mosdef_gomc.utils.gmso_equation_compare import (
        get_atom_type_expressions_and_scalars,
    )
except ImportError as e:
    print(f"ImportError: {e}")
    sys.exit(1)


class MockAtomType:
    def __init__(self, name):
        self.name = name


def run_repro():
    # Construct a dummy input that matches NONE of the forms (LJ, Mie, Exp6) and is NOT explicitly TABULATED
    # To avoid matching LJ/Mie/Exp6, I'll use a nonsense expression string.

    atom_types_dict = {
        "RES1": {
            "expression": "k * nonsense_variable",  # Standard forms expect specific symbols
            "atom_types": [MockAtomType("AT1")],
        }
    }

    print(
        "Running get_atom_type_expressions_and_scalars with undefined expression..."
    )
    result = get_atom_type_expressions_and_scalars(atom_types_dict)

    res_key = "RES1_AT1"
    if res_key in result:
        form = result[res_key].get("expression_form")
        scalar = result[res_key].get("expression_scalar")
        print(f"Expression Form: {form}")
        print(f"Expression Scalar: {scalar}")
    else:
        print("Result key not found.")


if __name__ == "__main__":
    run_repro()
