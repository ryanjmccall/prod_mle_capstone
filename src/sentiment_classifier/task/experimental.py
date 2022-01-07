import time
from typing import Tuple

import dask.array as da
from dask_ml.wrappers import ParallelPostFit
import prefect
from prefect import task


@task(name='evaluate_parallel_prediction', log_stdout=True)
def eval_parallel_pred(model, df, duplication=5) -> Tuple[float, int]:
    """Evaluate parallelized prediction using Dask."""
    logger = prefect.context.get('logger')

    X = df.features
    y = df.negativity
    parallel_model = ParallelPostFit(model, scoring='accuracy')
    parallel_model.fit(X, y)

    X_large = dupe_data(X, duplication)
    y_large = dupe_data(y, duplication)

    start = time.time()
    model.score(X_large, y_large)
    elapsed = time.time() - start
    predictions = X_large.shape[0]

    logger.info(f'elapsed  {elapsed:.3f} s')
    logger.info(f'items    {predictions:,}')
    logger.info(f'per item {elapsed / predictions * 1e6:.3f} us')
    return elapsed, predictions


def dupe_data(X, duplication: int):
    return da.concatenate([da.from_array(X.to_numpy(), chunks=X.shape)
                           for _ in range(duplication)])
