import os
import shutil
import sys
import uuid

from joblib import delayed, Parallel
from tqdm import tqdm as progressbar

from .io import save


def _create_job_id():
    return str(uuid.uuid4())


def split_into_chunks(input_list, chunk_size, chunk_dir, checkpoint=False):
    """
    Break the input data into smaller chunks, optionally saving each one to disk.

    Args:
        input_list: An input object that has a list-like interface (indexing and slicing).
        chunk_size: The maximum number of input items in each chunk.
        chunk_dir: The directory to save the checkpoints to.
        checkpoint: Whether to save each chunk to a file.

    Returns:
        A list of chunk objects with the following structure:
        {'index', 'data', 'input_filename', 'result_filename'}
    """

    if checkpoint and not os.path.exists(chunk_dir):
        os.mkdir(chunk_dir)

    chunks = [
        {
            'index': chunk_index,
            'data': input_list[start_index:start_index + chunk_size],
            'input_filename': os.path.join(chunk_dir, 'chunk-{:05d}-input.pickle'.format(chunk_index)),
            'result_filename': os.path.join(chunk_dir, 'chunk-{:05d}-output.pickle'.format(chunk_index)),
        }
        for chunk_index, start_index in enumerate(range(0, len(input_list), chunk_size))
    ]

    if checkpoint:
        for chunk in chunks:
            save(chunk['data'], chunk['input_filename'])

    return chunks


def map_embarrassingly_parallel(input_list, mapper, project, n_jobs=-1, chunk_size=-1,
                                checkpoint=False, cleanup=True, **kwargs):
    """
    Process items in a list in parallel (optionally, in smaller batches).

    Args:
        input_list: An input object that has a list-like interface (indexing and slicing).
        mapper: A function to apply to each item of the input list.
        project: An instance of pygoose project.
        n_jobs: The number of parallel processing jobs. -1 will use the number of CPUs on the system.
        chunk_size: The maximum number of input items in each chunk. -1 will store all data as a single chunk.
        checkpoint: Whether to save each chunk and its corresponding output to disk.
        cleanup: Whether to remove the chunk checkpoints from the disk after all chunks are processed.
        **kwargs: Additional keyword arguments to joblib.Parallel.

    Returns:
        A list representing the combined output from the mapper function called on all input items.
    """

    if chunk_size < 0:
        chunk_size = len(input_list)

    # Partition the data.
    job_id = _create_job_id()
    print('Creating job ID:', job_id)

    chunk_dir = os.path.join(project.temp_dir, job_id)
    chunks = split_into_chunks(input_list, chunk_size, chunk_dir, checkpoint)

    # The results will be collected here.
    # TODO: collecting lists like this may be memory inefficient. Perhaps we could use another callback function.
    combined_results = []

    # Process data one chunk at a time.
    for chunk in chunks:
        description = 'Chunk {}/{}'.format(chunk['index'] + 1, len(chunks))

        # Process each item in a chunk in parallel.
        chunk_result = Parallel(n_jobs=n_jobs, **kwargs)(
            delayed(mapper)(input_item)
            for input_item in progressbar(
                chunk['data'],
                desc=description,
                total=len(chunk['data']),
                file=sys.stdout,
            )
        )
        if checkpoint:
            save(chunk_result, chunk['result_filename'])

        combined_results.extend(chunk_result)

    # Remove the temporary files.
    if checkpoint and cleanup:
        shutil.rmtree(chunk_dir)

    return combined_results


def map_job_per_chunk(input_list, mapper, project, n_jobs=-1, chunk_size=-1, **kwargs):
    """
    Split the data into chunks and process each chunk in its own thread.

    Args:
        input_list: An input object that has a list-like interface (indexing and slicing).
        mapper: A function to apply to each chunk.
        project: An instance of pygoose project.
        n_jobs: The number of parallel processing jobs. -1 will use the number of CPUs on the system.
        chunk_size: The maximum number of input items in each chunk. -1 will store all data as a single chunk.
        **kwargs: Additional keyword arguments to joblib.Parallel.

    Returns:
        A list representing the combined output from the mapper function called on all input items of each chunk.
    """

    if chunk_size < 0:
        chunk_size = len(input_list)

    job_id = _create_job_id()
    print('Creating job ID:', job_id)

    chunk_dir = os.path.join(project.temp_dir, job_id)
    chunks = split_into_chunks(input_list, chunk_size, chunk_dir)

    combined_results = Parallel(n_jobs=-n_jobs, **kwargs)(
        delayed(mapper)(chunk['data'])
        for chunk in progressbar(
            chunks,
            desc='Chunks',
            total=len(chunks),
            file=sys.stdout,
        )
    )

    if os.path.exists(chunk_dir):
        shutil.rmtree(chunk_dir)

    return combined_results
