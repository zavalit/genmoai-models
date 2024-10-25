import ray

from mochi_preview.t2v_synth_mochi import T2VSynthMochiModel


def noexcept(f):
    try:
        return f()
    except:
        pass


class MochiWrapper:
    def __init__(self, *, num_workers, **actor_kwargs):
        super().__init__()
        #ray.init(num_cpus=2, num_gpus=0)

        RemoteClass = ray.remote(T2VSynthMochiModel)
        self.workers = [
            RemoteClass.options(num_gpus=0).remote(
                device_id=0, world_size=num_workers, local_rank=i, **actor_kwargs
            )
            for i in range(num_workers)
        ]
        # Ensure the __init__ method has finished on all workers
        for worker in self.workers:
            ray.get(worker.__ray_ready__.remote())
        self.is_loaded = True

    def __call__(self, args):
        work_refs = [
            worker.run.remote(args, i == 0) for i, worker in enumerate(self.workers)
        ]

        try:
            for result in work_refs[0]:
                yield ray.get(result)

            # Handle the (very unlikely) edge-case where a worker that's not the 1st one
            # fails (don't want an uncaught error)
            for result in work_refs[1:]:
                ray.get(result)
        except Exception as e:
            # Get exception from other workers
            for ref in work_refs[1:]:
                noexcept(lambda: ray.get(ref))
            raise e
