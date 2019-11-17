"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from playground import maybe_initialize_process
from playground.samplers.sampler import Sampler
from playground.samplers.sequential_sampler import SequentialSampler
import multiprocessing as m


def create_sampler_process(
        env,
        agent,
        max_path_length,
        max_buffer_size=100
):
    # create a new process to handle sampling trajectories
    set_weights_input_queue = m.Queue(maxsize=max_buffer_size)
    collect_input_queue = m.Queue(maxsize=max_buffer_size)
    collect_output_queue = m.Queue(maxsize=max_buffer_size)

    m.Process(target=process_function, args=(
        env,
        agent,
        max_path_length,
        set_weights_input_queue,
        collect_input_queue,
        collect_output_queue)).start()

    return (set_weights_input_queue,
            collect_input_queue,
            collect_output_queue)


def process_function(
        env,
        agent,
        max_path_length,
        set_weights_input_queue,
        collect_input_queue,
        collect_output_queue
):
    # initialize tensorflow and the multiprocessing interface
    maybe_initialize_process(use_gpu=False)

    # create a sampler instance within this thread
    sequential_sampler = SequentialSampler(
        env,
        agent,
        max_path_length=max_path_length)

    # loop until a termination signal is given to this thread
    is_finished = False
    while not is_finished:

        # set the weights of the policy
        if not set_weights_input_queue.empty():
            sequential_sampler.set_weights(set_weights_input_queue.get())

        # collect k paths of samples and pass to the main process
        if not collect_input_queue.empty():
            (min_num_steps_to_collect,
             deterministic,
             save_data, render, render_kwargs) = collect_input_queue.get()

            # push results back to the main thread
            collect_output_queue.put(
                sequential_sampler.collect(
                    min_num_steps_to_collect,
                    deterministic=deterministic,
                    keep_data=save_data, render=render, render_kwargs=render_kwargs))


class ParallelSampler(Sampler):

    def __init__(
            self,
            env,
            agent,
            max_path_length=1000,
            num_workers=1,
            max_buffer_size=100
    ):
        # create several processes in which sampling will occur
        self.num_workers = num_workers
        self.set_weights_input_queues = []
        self.collect_input_queues = []
        self.collect_output_queues = []

        # for each process keep track of input and output queues
        for i in range(num_workers):
            (set_weights_input_queue,
             collect_input_queue,
             collect_output_queue) = create_sampler_process(
                env,
                agent,
                max_path_length,
                max_buffer_size=max_buffer_size)

            self.set_weights_input_queues.append(set_weights_input_queue)
            self.collect_input_queues.append(collect_input_queue)
            self.collect_output_queues.append(collect_output_queue)

    def set_weights(
            self,
            weights
    ):
        # set the weights for each agent in each process
        for q in self.set_weights_input_queues:
            q.put(weights)

    def collect(
            self,
            min_num_steps_to_collect,
            deterministic=False,
            keep_data=False,
            render=False,
            render_kwargs=None,
            workers_to_use=1
    ):
        # pass arguments for path collection to the workers
        workers_to_use = min(workers_to_use, self.num_workers)
        for i in range(workers_to_use):
            target_count = min_num_steps_to_collect // workers_to_use
            if (min_num_steps_to_collect % workers_to_use) - i > 0:
                target_count += 1
            self.collect_input_queues[i].put((
                target_count, deterministic, keep_data, render, render_kwargs))

        # return paths from the workers into the main process
        results = []
        while len(results) < workers_to_use:
            for q in self.collect_output_queues:
                if not q.empty():
                    results.append(q.get())

        # combine the paths returns and steps from each of the remote samplers
        paths = [path for item in results for path in item[0]]
        returns = [path_return for item in results for path_return in item[1]]
        return paths, returns, sum([item[2] for item in results])
