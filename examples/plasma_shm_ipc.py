from functools import partial

from pyarrow import plasma  # type: ignore

from mosec import Server, Worker
from mosec.errors import ValidationError
from mosec.plugins import PlasmaShmWrapper


class DataProducer(Worker):
    def forward(self, data: dict) -> bytes:
        try:
            data_bytes = b"a" * data["size"]
        except KeyError as err:
            raise ValidationError(err)
        return data_bytes


class DataConsumer(Worker):
    def forward(self, data: bytes) -> dict:
        return {"ipc test data length": len(data)}


if __name__ == "__main__":
    """
    We start a subprocess for the plasma server, and pass the path
    to the plasma client which serves as the shm wrapper.
    We also register the plasma server process as a daemon, so
    that when it exits the service is able to gracefully shutdown
    and restarted by the orchestrator.
    """
    # 200 Mb store, adjust the size according to your requirement
    with plasma.start_plasma_store(plasma_store_memory=200 * 1000 * 1000) as (
        shm_path,
        shm_process,
    ):
        server = Server(
            ipc_wrapper=partial(  # defer the wrapper init to worker processes
                PlasmaShmWrapper,
                shm_path=shm_path,
            )
        )
        server.register_daemon("plasma_server", shm_process)
        server.append_worker(DataProducer, num=2)
        server.append_worker(DataConsumer, num=2)
        server.run()
