# contains solver options for different CasADi solvers


class SolverOptions:
    solver_options = {}
    solver_options["ipopt"] = {
        "expand": True,
        "show_eval_warnings": True,
        "warn_initial_bounds": True,
        "print_time": False,
        "record_time": True,
        "bound_consistency": True,
        "calc_lam_x": True,
        "calc_lam_p": False,
        "ipopt": {
            # "linear_solver": "ma97",
            # "linear_system_scaling": "mc19",
            # "nlp_scaling_method": "equilibration-based",
            "max_iter": 500,
            "sb": "yes",
            "print_level": 0,
        },
    }

    solver_options["qrqp"] = {
        "expand": True,
        "print_time": False,
        "record_time": True,
        "error_on_fail": False,
        "print_info": False,
        "print_iter": False,
        "print_header": False,
        "max_iter": 2000,
    }

    solver_options["qpoases"] = {
        "print_time": False,
        "record_time": True,
        "error_on_fail": False,
        "printLevel": "none",
    }

    @staticmethod
    def get_solver_options(solver_name: str) -> dict:
        if solver_name not in SolverOptions.solver_options:
            raise ValueError(f"Solver {solver_name} not supported.")
        return SolverOptions.solver_options[solver_name]
