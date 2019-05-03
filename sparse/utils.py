import logging


def set_batch_size(input_dict, batch_size):
    (previous_batch_size, new_batch_size) = batch_size
    if previous_batch_size <= new_batch_size:
        return input_dict
    assert new_batch_size < previous_batch_size

    def f(k, v):
        if v.shape[0] == previous_batch_size:
            logging.info(
                "Changing batch size of %s (%s) from %s to %s",
                k,
                v.shape,
                previous_batch_size,
                new_batch_size,
            )
            return v[:new_batch_size]
        logging.info("Preserving batch size of %s (%s)", k, v.shape)
        return v

    return {k: f(k, v) for k, v in input_dict.items()}