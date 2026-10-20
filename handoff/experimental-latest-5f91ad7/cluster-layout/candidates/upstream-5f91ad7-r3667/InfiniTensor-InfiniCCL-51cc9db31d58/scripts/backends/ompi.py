import os

from backends.mpi_base import BaseMpiBackend


class OmpiBackend(BaseMpiBackend):
    def get_hostfile_line(self, ip, slots):
        return f"{ip} slots={slots}\n"

    def get_base_mpi_args(self, hostfile_path, total_slots):
        cmd = ["mpirun", "--hostfile", hostfile_path, "-np", str(total_slots)]

        # Safety: OpenMPI refuses to run as root unless explicitly told.
        if os.getuid() == 0:
            cmd.append("--allow-run-as-root")

        return cmd

    def get_env_args(self, env_key):
        return ["-x", env_key]
