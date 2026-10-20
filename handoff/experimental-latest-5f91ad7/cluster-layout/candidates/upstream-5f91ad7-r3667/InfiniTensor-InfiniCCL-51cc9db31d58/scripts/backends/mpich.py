from backends.mpi_base import BaseMpiBackend


class MpichBackend(BaseMpiBackend):
    def get_hostfile_line(self, ip, slots):
        return f"{ip}:{slots}\n"

    def get_base_mpi_args(self, hostfile_path, total_slots):
        # MPICH uses `-f` for hostfile, `-n` for slot allocations, no root check needed.
        return ["mpirun", "-f", hostfile_path, "-n", str(total_slots)]

    def get_env_args(self, env_key):
        # MPICH/Hydra utilizes global `genv` flags.
        return ["-genvlist", env_key]
