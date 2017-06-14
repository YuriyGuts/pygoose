import os
import shutil
import sys
import uuid

from joblib import delayed, Parallel
from tqdm import tqdm as progressbar

from .io import save


def _create_job_id():
    return str(uuid.uuid4())


def split_into_batches(input_list, batch_size, batch_storage_dir, checkpoint=False):
    """
    Break the input data into smaller batches, optionally saving each one to disk.

    Args:
        input_list: An input object that has a list-like interface (indexing and slicing).
        batch_size: The maximum number of input items in each batch.
        batch_storage_dir: The directory to save the checkpoints to.
        checkpoint: Whether to save each batch to a file.

    Returns:
        A list of batch objects with the following structure:
        {'index', 'data', 'input_filename', 'result_filename'}
    """

    if checkpoint and not os.path.exists(batch_storage_dir):
        os.mkdir(batch_storage_dir)

    batches = [
        {
            'index': batch_index,
            'data': input_list[start_index:start_index + batch_size],
            'input_filename': os.path.join(batch_storage_dir, 'batch-{:05d}-input.pickle'.format(batch_index)),
            'result_filename': os.path.join(batch_storage_dir, 'batch-{:05d}-output.pickle'.format(batch_index)),
        }
        for batch_index, start_index in enumerate(range(0, len(input_list), batch_size))
    ]

    if checkpoint:
        for batch in batches:
            save(batch['data'], batch['input_filename'])

    return batches


def map_embarrassingly_parallel(input_list, mapper, project, n_jobs=-1, batch_size=-1,
                                checkpoint=False, cleanup=True, **kwargs):
    """
    Process items in a list in parallel (optionally, one smaller batch at a time).

    Args:
        input_list: An input object that has a list-like interface (indexing and slicing).
        mapper: A function to apply to each item of the input list.
        project: An instance of pygoose project.
        n_jobs: The number of parallel processing jobs. -1 will use the number of CPUs on the system.
        batch_size: The maximum number of input items in each batch. -1 will store all data as a single batch.
        checkpoint: Whether to save each batch and its corresponding output to disk.
        cleanup: Whether to remove the batch checkpoints from the disk after all batches are processed.
        **kwargs: Additional keyword arguments to joblib.Parallel.

    Returns:
        A list representing the combined output from the mapper function called on all input items.
    """

    if batch_size < 0:
        batch_size = len(input_list)

    # Partition the data.
    job_id = _create_job_id()
    print('Creating job ID:', job_id)

    batch_storage_dir = os.path.join(project.temp_dir, job_id)
    batches = split_into_batches(input_list, batch_size, batch_storage_dir, checkpoint)

    # The results will be collected here.
    # TODO: collecting lists like this may be memory inefficient. Perhaps we could use another callback function.
    combined_results = []

    # Process data one batch at a time.
    for batch in batches:
        description = 'Batch {}/{}'.format(batch['index'] + 1, len(batches))

        # Process each item in the batch in parallel.
        batch_result = Parallel(n_jobs=n_jobs, **kwargs)(
            delayed(mapper)(input_item)
            for input_item in progressbar(
                batch['data'],
                desc=description,
                total=len(batch['data']),
                file=sys.stdout,
            )
        )
        if checkpoint:
            save(batch_result, batch['result_filename'])

        combined_results.extend(batch_result)

    # Remove the temporary files.
    if checkpoint and cleanup:
        shutil.rmtree(batch_storage_dir)

    return combined_results


def map_batch_parallel(input_list, batch_size, item_mapper=None, batch_mapper=None, flatten=True, n_jobs=-1, **kwargs):
    """
    Split the data into batches and process each batch in its own thread.

    Args:
        input_list: An input object that has a list-like interface (indexing and slicing).
        item_mapper: (optional) A function to apply to each item in the batch.
        batch_mapper: (optional) A function to apply to each batch. Either item_mapper or batch_mapper must be set.
        flatten: Whether to unwrap individual batch results or keep them grouped by batch.
        n_jobs: The number of parallel processing jobs. -1 will use the number of CPUs on the system.
        batch_size: The maximum number of input items in each batch. -1 will store all data as a single batch.
        **kwargs: Additional keyword arguments to joblib.Parallel.

    Returns:
        A list representing the combined output from the mapper function called on all input items of each batch.
    """

    # We must specify either how to process each batch or how to process each item.
    if item_mapper is None and batch_mapper is None:
        raise ValueError('You should specify either batch_mapper or item_mapper.')

    if batch_mapper is None:
        batch_mapper = _default_batch_mapper

    batches = split_into_batches(input_list, batch_size, batch_storage_dir='')
    all_batch_results = Parallel(n_jobs=n_jobs, **kwargs)(
        delayed(batch_mapper)(batch['data'], item_mapper)
        for batch in progressbar(
            batches,
            desc='Batches',
            total=len(batches),
            file=sys.stdout,
        )
    )

    # Unwrap the individual batch results if necessary.
    if flatten:
        final_result = []
        for batch_result in all_batch_results:
            final_result.extend(batch_result)
    else:
        final_result = all_batch_results

    return final_result


def _default_batch_mapper(batch, item_mapper):
    return [item_mapper(item) for item in batch]
