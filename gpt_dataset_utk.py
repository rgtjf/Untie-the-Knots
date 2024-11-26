import logging
import string
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy
import numpy as np
import torch
from megatron.core.datasets.gpt_dataset import (
    GPTDataset,
    GPTDatasetConfig,
    _get_ltor_masks_and_position_ids,
)

logger = logging.getLogger(__name__)


@dataclass
class GPTDatasetUtKConfig(GPTDatasetConfig):
    chunk_tokens: list[int] = None
    """List of tokens, where the length should be `n_chunks * 2 - 2`.
    It includes the end token of the first segment, the start and end tokens of intermediate segments,
    and the start token of the last segment, following the pattern: seg_1_1, seg_1_2, seg_2_1, seg_2_2, ..., 
    seg_(n-1)_1, seg_(n-1)_2."""

    corruption_rng_seed: int = 47
    """Seed for the random number generator used in corruption."""

    use_length_sampler: bool = False
    """Use length sampler for corruption"""

    chunk_probs_cfg: tuple = None
    """chunk_probs_cfg, format ([0.7, 0.3], [[1, 0, 0], [0, 0, 1.0]])
    We use hop4 as chunking to spans."""

    skywork_interleave: bool = False

    preserve_partial_order: bool = False

    add_sort_task: bool = False

    # <D> </D> <S> <s> </S>
    # 131403, 131404, 13105, 131406, 131407
    sort_tokens: list[int] = None

    def __post_init__(self) -> None:
        super().__post_init__()

        assert all(self.chunk_tokens), "chunk_tokens is not set!"
        chunk_num = len(self.chunk_probs_cfg[1][0])
        assert (
            len(self.chunk_tokens) == (chunk_num - 1) * 2
        ), f"The length of chunk_tokens should be {(chunk_num - 1) * 2}."


def create_rng_with_seed_and_idx(seed, idx):
    """Create a random number generator with a combined seed from the given seed and index.

    Args:
        seed (int): The base seed.
        idx (int): The index to combine with the seed.

    Returns:
        numpy.random.Generator: A random number generator initialized with the combined seed.
    """
    combined_seed = seed * 100000 + idx
    mt19937 = numpy.random.MT19937(combined_seed)
    rng = numpy.random.Generator(mt19937)
    return rng


def interleaved_skywork(arr, rng):
    """
    Do interleaved just like skywork do

    Args:
        arr (list of list): Each inner list contains the parts of one sample.
        rng (np.random.Generator): Useless in skywork mode

    Returns:
        list: A shuffled flattened list.
    """

    # Calculate the number of parts in each sample
    # Decrement the count of the last part of the last sample
    parts_num = [len(parts) for parts in arr]

    # Initialize the results list with None
    results_num = sum(parts_num)
    results = [None for _ in range(results_num)]

    counter = 0
    for length in range(max(parts_num)):
        for sample_idx in range(len(arr)):
            if length < len(arr[sample_idx]):
                results[counter] = (sample_idx, length)
                counter += 1

    # Combine the shuffled parts and add the last part of the last sample
    shuffle_arr = [arr[sample_idx][part_idx] for sample_idx, part_idx in results]
    return shuffle_arr


def interleaved_skywork_fast(arr, rng):
    """
    Do interleaved just like skywork does.

    Args:
        arr (list of list): Each inner list contains the parts of one sample.
        rng (np.random.Generator): Useless in skywork mode

    Returns:
        list: A shuffled flattened list.
    """
    # Calculate the number of parts in each sample
    parts_num = np.array([len(parts) for parts in arr])

    # Determine the maximum number of parts
    max_parts = parts_num.max()

    # Initialize the results array with None
    indices = np.full((len(arr), max_parts), fill_value=-1, dtype=int)

    for sample_idx, length in enumerate(parts_num):
        indices[sample_idx, :length] = np.arange(length)

    # Flatten the indices while preserving the order
    valid_indices = indices[indices != -1]
    sample_indices = np.repeat(np.arange(len(arr)), parts_num)

    # Combine the parts
    shuffle_arr = [
        arr[sample_idx][part_idx] for sample_idx, part_idx in zip(sample_indices, valid_indices)
    ]
    return shuffle_arr


def shuffle_preserve_partial_order_multi_hop(parts_num, rng, preserve_partial_order=False):
    """Shuffles parts of each sample in a nested list, preserving the partial order,
    except for the last part of the last sample, which remains at the end.

    Args:
        parts_num (list of int): List containing the number of parts for each sample.
        rng (np.random.Generator): A random number generator instance.
        preserve_partial_order (bool, optional): Whether to preserve the partial order. Defaults to False.

    Returns:
        tuple: A tuple containing the shuffled adjacency matrix and a dictionary mapping sample indices to part indices.
    """
    # Decrement the count of the last part of the last sample
    parts_num[-1] -= 1

    total_parts = sum(parts_num)
    shuffled_arr = [None] * total_parts
    indices_in_arr = np.arange(total_parts)
    rng.shuffle(indices_in_arr)

    start = 0
    sample_to_part_indices_map = {}
    for sample_idx, num_parts in enumerate(parts_num):
        if num_parts == 0:
            continue
        part_indices_in_arr = indices_in_arr[start: start + num_parts]
        if preserve_partial_order:
            part_indices_in_arr.sort()
        sample_to_part_indices_map[sample_idx] = part_indices_in_arr.tolist()
        for part_idx, index_in_arr in enumerate(part_indices_in_arr):
            shuffled_arr[index_in_arr] = (sample_idx, part_idx)
        start += num_parts

    sample_idx, part_idx = len(parts_num) - 1, parts_num[-1]
    shuffled_arr.append((sample_idx, part_idx))
    if sample_idx not in sample_to_part_indices_map:
        sample_to_part_indices_map[sample_idx] = [part_idx]
    else:
        sample_to_part_indices_map[sample_idx].append(part_idx)
    return shuffled_arr, sample_to_part_indices_map


def generate_random_string(length=6, rng=None):
    """
    Generates a random string of a specified length using the provided random number generator.

    Args:
        length (int, optional): Length of the generated string. Defaults to 6.
        rng (np.random.Generator, optional): A random number generator instance. Defaults to None.

    Returns:
        str: A randomly generated string.
    """
    if rng is None:
        rng = np.random.default_rng()

    chars = list(string.ascii_letters + string.digits)
    return ''.join(rng.choice(chars, size=length))


def process_sorted_samples(
    segmented_samples, shuffled_arr, sample_to_part_indices_map, tokenizer, rng, sort_tokens
):
    """
    Process the shuffled samples, generate doc IDs, and handle the sorting task.

    Args:
        segmented_samples (list): List of segmented samples.
        shuffled_arr (list): Shuffled array containing sample and part indices.
        sample_to_part_indices_map: dictionary mapping sample indices to part indices.
        tokenizer: Tokenizer instance for generating random strings.
        rng (np.random.Generator): A random number generator instance.
        sort_tokens (list[int]): Tokens for sorting task.

    Returns:
        list: A list of processed and concatenated samples.
    """
    if tokenizer is None:
        raise ValueError("tokenizer must be provided if add_sort_task is True.")
    if sort_tokens is None or len(sort_tokens) != 5:
        raise ValueError("sort_tokens must be a list of exactly 5 tokens.")

    D_start_token, D_end_token, S_start_token, S_mid_token, S_end_token = sort_tokens

    # Generate document IDs for each document in the shuffled array
    doc_ids = [tokenizer.tokenize(generate_random_string(6, rng)) for _ in range(len(shuffled_arr))]

    shuffled_samples = []
    for idx, (sample_idx, part_idx) in enumerate(shuffled_arr):
        # recover the original order
        part_indices = sample_to_part_indices_map[sample_idx]

        if len(part_indices) == 1:
            # If there's only one part, we don't apply the sorting task
            shuffled_samples.append(segmented_samples[sample_idx][part_idx])
        else:
            # <D> doc_id </D> text
            context = np.concatenate(
                [
                    [D_start_token],
                    doc_ids[idx],
                    [D_end_token],
                    segmented_samples[sample_idx][part_idx],
                ]
            )

            # <S> doc_id <s> doc_id </S>
            # We add the summary to the last part.
            if idx == max(part_indices):
                summary = [S_start_token]
                for _id, _part_index in enumerate(part_indices):
                    if _id != 0:
                        summary.append(S_mid_token)
                    summary.extend(doc_ids[_part_index])
                summary.append(S_end_token)
                # Concatenate context and summary when needed.
                context = np.concatenate([context, summary])

            shuffled_samples.append(context)

    return shuffled_samples


def denoise_text_multi_hop(
    text: numpy.ndarray,
    rng: numpy.random.Generator,
    eod_token: int = 0,
    chunk_tokens: list[int] = None,
    chunk_probs: list[float] = None,
    use_length_sampler: bool = False,
    skywork_interleave: bool = False,
    preserve_partial_order: bool = False,
    tokenizer: "cybertron.tokenizer.tokenizer.HFTokenizer" = None,
    add_sort_task: bool = False,
    sort_tokens: list[int] = None,
) -> numpy.ndarray:
    """Apply denoising corruption to a text sequence.

    Args:
        text (numpy.ndarray): The text sequence to denoise.
        rng (numpy.random.Generator): A random number generator.
        eod_token (int, optional): The end-of-document token. Defaults to 0.
        chunk_tokens: List of tokens, where the length should be `n_chunks * 2 - 2`. following the pattern: seg_1_1,
        seg_1_2, seg_2_1, seg_2_2,
        chunk_probs: list of probabilities, where the length is 3.
        use_length_sampler (bool, optional): Whether to use a length sampler for corruption. Defaults to False.
        skywork_interleave (bool, optional): Whether to use skywork interleaved mode. Defaults to False.
        preserve_partial_order (bool, optional): Whether to preserve partial order. Defaults to False.
        tokenizer ("cybertron.tokenizer.tokenizer.HFTokenizer", optional): Tokenizer instance for generating random strings.
        add_sort_task (bool, optional): Whether to add a sorting task for the segmented samples. Defaults to False.
        sort_tokens (list[int], optional): Tokens for sorting task, should contain exactly 5 tokens.

    Returns:
        numpy.ndarray: The denoised text sequence.
    """
    sequence_length = text.shape[0]

    # 1. segment_samples
    eod_indices = numpy.where(text == eod_token)[0]
    eod_indices = [0] + (eod_indices + 1).tolist()
    if eod_indices[-1] != sequence_length:
        eod_indices = eod_indices + [sequence_length]
    samples = [text[start:end] for start, end in zip(eod_indices[:-1], eod_indices[1:])]

    # 2. segment_sample
    segmented_samples = []
    for sample in samples:
        # 2.1 calculate_split_positions
        n_chunks = rng.choice(len(chunk_probs), 1, p=chunk_probs)[0] + 1
        if n_chunks == 4:
            split_positions = list(range(1024, len(sample), 1024))
        elif len(sample) < n_chunks:
            split_positions = []
        else:
            if use_length_sampler:
                split_positions = rng.choice(
                    range(1, len(sample)), n_chunks - 1, replace=False
                ).tolist()
                split_positions = sorted(split_positions)
            else:
                split_positions = [
                    len(sample) // n_chunks * chunk_id for chunk_id in range(1, n_chunks)
                ]
        split_positions = [0] + split_positions + [len(sample)]
        # 2.2 segment_sample
        sample_parts = []
        chunk_id = 0
        for start, end in zip(split_positions[:-1], split_positions[1:]):
            sample_part = sample[start:end]
            if skywork_interleave:
                pass
            elif n_chunks == 4:
                if start != 0:
                    add_tokens = (
                        [chunk_tokens[0]] + tokenizer.tokenize(str(chunk_id)) + [chunk_tokens[1]]
                    )
                    sample_part = numpy.insert(sample_part, 0, add_tokens)
                if end != len(sample):
                    add_tokens = (
                        [chunk_tokens[2]] + tokenizer.tokenize(str(chunk_id)) + [chunk_tokens[3]]
                    )
                    sample_part = numpy.append(sample_part, add_tokens)
            else:
                if start != 0:
                    sample_part = numpy.insert(sample_part, 0, chunk_tokens[2 * chunk_id - 1])
                if end != len(sample):
                    sample_part = numpy.append(sample_part, chunk_tokens[2 * chunk_id])
            sample_parts.append(sample_part)
            chunk_id += 1
        segmented_samples.append(sample_parts)

    # 3. interleaved or shuffled
    # skywork_interleave used for baseline
    if skywork_interleave:
        shuffled_samples = interleaved_skywork(segmented_samples, rng)
    else:
        # 3.1 shuffle
        parts_num = [len(parts) for parts in segmented_samples]
        shuffled_arr, sample_to_part_indices_map = shuffle_preserve_partial_order_multi_hop(
            parts_num, rng, preserve_partial_order
        )
        # 3.2 context and summary
        if not add_sort_task:
            shuffled_samples = [
                segmented_samples[sample_idx][part_idx] for sample_idx, part_idx in shuffled_arr
            ]
        else:
            shuffled_samples = process_sorted_samples(
                segmented_samples,
                shuffled_arr,
                sample_to_part_indices_map,
                tokenizer,
                rng,
                sort_tokens,
            )
    tokens = numpy.concatenate(shuffled_samples)
    tokens = tokens[:sequence_length]
    return numpy.array(tokens, dtype=numpy.int64)


class GPTDatasetUtK(GPTDataset):

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Abstract method implementation

        Args:
            idx (int): The index into the dataset

        Returns:
            Dict[str, torch.Tensor]: The text ids wrapped in a dictionary
        """
        text, _ = self._query_document_sample_shuffle_indices(idx)

        rng = create_rng_with_seed_and_idx(self.config.corruption_rng_seed, idx)

        chunk_probs = rng.choice(
            self.config.chunk_probs_cfg[1], 1, p=self.config.chunk_probs_cfg[0]
        )[0]

        text = denoise_text_multi_hop(
            text,
            rng,
            self.config.tokenizer.eod,
            self.config.chunk_tokens,
            chunk_probs,
            self.config.use_length_sampler,
            self.config.skywork_interleave,
            self.config.preserve_partial_order,
            self.config.tokenizer,
            self.config.add_sort_task,
            self.config.sort_tokens,
        )

        text = torch.from_numpy(text).long()

        # the same as before
        labels = text[1:].contiguous()
        tokens = text[:-1].contiguous()

        assert not torch.any(
            tokens >= self.config.tokenizer.vocab_size
        ), "An input token is out of bounds of the tokenizer vocabulary"

        # caannot be cached
        attention_mask, loss_mask, position_ids = _get_ltor_masks_and_position_ids(
            tokens,
            self.config.tokenizer.eod,
            self.config.reset_position_ids,
            self.config.reset_attention_mask,
            self.config.eod_mask_loss,
            self.config.create_attention_mask,
        )

        # Mask loss for the segment tokens
        if self.config.eod_mask_loss:
            for chunk_token in self.config.chunk_tokens:
                loss_mask[tokens == chunk_token] = 0.0
                loss_mask[labels == chunk_token] = 0.0

            if self.config.add_sort_task:
                # mask token: <D> docid </D>
                # mask label: <D>
                # mask token: </S>
                # mask label: <S>
                D_start_token, D_end_token, S_start_token, S_mid_token, S_end_token = (
                    self.config.sort_tokens
                )
                start_positions = np.where(tokens == D_start_token)[0]
                end_positions = np.where(tokens == D_end_token)[0]
                if len(end_positions) == len(start_positions) - 1:
                    end_positions = np.append(end_positions, tokens.shape[0] - 1)
                for start, end in zip(start_positions, end_positions):
                    loss_mask[start: end + 1] = 0.0
                loss_mask[labels == D_start_token] = 0.0
                loss_mask[tokens == S_end_token] = 0.0
                loss_mask[labels == S_start_token] = 0.0

        if self.config.create_attention_mask:
            return {
                "tokens": tokens,
                "labels": labels,
                "attention_mask": attention_mask,
                "loss_mask": loss_mask,
                "position_ids": position_ids,
            }
        else:
            return {
                "tokens": tokens,
                "labels": labels,
                "loss_mask": loss_mask,
                "position_ids": position_ids,
            }

    @staticmethod
    def _key_config_attributes() -> List[str]:
        """Return all config attributes which contribute to uniquely identifying the dataset.

        These attributes will be used to build a uniquely identifying string and MD5 hash which
        will be used to cache/load dataset resources from run to run.

        Returns:
            List[str]: The key config attributes
        """
        origin_key_config_attributes = GPTDataset._key_config_attributes()
        return origin_key_config_attributes
