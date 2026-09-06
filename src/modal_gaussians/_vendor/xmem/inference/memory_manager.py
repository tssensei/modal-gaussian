import warnings

import torch
from modal_gaussians._vendor.xmem.inference.kv_memory_store import KeyValueMemoryStore
from modal_gaussians._vendor.xmem.model.memory_util import *


class MemoryManager:
    """
    Manages all three memory stores and the transition between working/long-term memory
    """

    def __init__(self, config):
        self.hidden_dim = config["hidden_dim"]
        self.top_k = config["top_k"]

        self.query_chunk_size = self._query_chunk_size(config)

        self.enable_long_term = config["enable_long_term"]
        self.enable_long_term_usage = config["enable_long_term_count_usage"]
        if self.enable_long_term:
            self.max_mt_frames = config["max_mid_term_frames"]
            self.min_mt_frames = config["min_mid_term_frames"]
            self.num_prototypes = config["num_prototypes"]
            self.max_long_elements = config["max_long_term_elements"]

        # dimensions will be inferred from input later
        self.CK = self.CV = None
        self.H = self.W = None

        # The hidden state will be stored in a single tensor for all objects
        # B x num_objects x CH x H x W
        self.hidden = None

        self.work_mem = KeyValueMemoryStore(count_usage=self.enable_long_term)
        if self.enable_long_term:
            self.long_mem = KeyValueMemoryStore(count_usage=self.enable_long_term_usage)

        self.reset_config = True

    def update_config(self, config):
        self.query_chunk_size = self._query_chunk_size(config)
        self.reset_config = True
        self.hidden_dim = config["hidden_dim"]
        self.top_k = config["top_k"]

        assert self.enable_long_term == config["enable_long_term"], "cannot update this"
        assert (
            self.enable_long_term_usage == config["enable_long_term_count_usage"]
        ), "cannot update this"

        self.enable_long_term_usage = config["enable_long_term_count_usage"]
        if self.enable_long_term:
            self.max_mt_frames = config["max_mid_term_frames"]
            self.min_mt_frames = config["min_mid_term_frames"]
            self.num_prototypes = config["num_prototypes"]
            self.max_long_elements = config["max_long_term_elements"]

    def _readout(self, affinity, v):
        # this function is for a single object group
        return v @ affinity

    @staticmethod
    def _query_chunk_size(config):
        size = config.get("query_chunk_size", 256)
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError("XMem query_chunk_size must be a nonnegative integer (0 = dense)")
        return size

    def match_memory(self, query_key, selection):
        """Bound affinity allocations without truncating the memory bank.

        Each query retains the original global top-k and softmax. Usage is
        accumulated over chunks and age advances once per video frame, retaining
        the original consolidation time scale. The caller uses inference_mode.
        """
        h, w = query_key.shape[-2:]
        query_key = query_key.flatten(start_dim=2)
        selection = selection.flatten(start_dim=2) if selection is not None else None
        long_mem_size = 0
        if self.enable_long_term and self.long_mem.engaged():
            long_mem_size = self.long_mem.size
            memory_key = torch.cat([self.long_mem.key, self.work_mem.key], -1)
            shrinkage = torch.cat([self.long_mem.shrinkage, self.work_mem.shrinkage], -1)
            all_memory_value = [
                torch.cat([self.long_mem.value[gi], gv], -1)
                if gi < self.long_mem.num_groups else gv
                for gi, gv in enumerate(self.work_mem.value)
            ]
        else:
            memory_key, shrinkage = self.work_mem.key, self.work_mem.shrinkage
            all_memory_value = self.work_mem.value

        chunk_size = self.query_chunk_size or h * w
        output = all_memory_value[0].new_empty(
            (sum(value.shape[0] for value in all_memory_value), self.CV, h * w)
        )
        usage_sum = None
        for start in range(0, h * w, chunk_size):
            stop = min(start + chunk_size, h * w)
            readout, usage = self._match_memory_chunk(
                query_key[:, :, start:stop],
                selection[:, :, start:stop] if selection is not None else None,
                memory_key, shrinkage, all_memory_value, long_mem_size,
            )
            output[:, :, start:stop] = readout
            if usage is not None:
                if usage_sum is None:
                    usage_sum = usage
                else:
                    usage_sum.add_(usage)

        if usage_sum is not None:
            self.work_mem.update_usage(usage_sum[:, long_mem_size:].flatten())
            if long_mem_size and self.enable_long_term_usage:
                self.long_mem.update_usage(usage_sum[:, :long_mem_size].flatten())
        return output.view(output.shape[0], self.CV, h, w)

    def _match_memory_chunk(self, query_key, selection, memory_key, shrinkage,
                            all_memory_value, long_mem_size):
        """Original group-specific equations applied to one query slice."""
        num_groups = self.work_mem.num_groups
        usage = None
        similarity = get_similarity(memory_key, shrinkage, query_key, selection)
        if long_mem_size:
            work_mem_similarity = similarity[:, long_mem_size:]
            long_mem_similarity = similarity[:, :long_mem_size]

            # get the usage with the first group
            # the first group always have all the keys valid
            affinity, usage = do_softmax(
                torch.cat(
                    [
                        long_mem_similarity[:, -self.long_mem.get_v_size(0) :],
                        work_mem_similarity,
                    ],
                    1,
                ),
                top_k=self.top_k,
                inplace=True,
                return_usage=True,
            )
            affinity = [affinity]

            # compute affinity group by group as later groups only have a subset of keys
            for gi in range(1, num_groups):
                if gi < self.long_mem.num_groups:
                    # merge working and lt similarities before softmax
                    affinity_one_group = do_softmax(
                        torch.cat(
                            [
                                long_mem_similarity[:, -self.long_mem.get_v_size(gi) :],
                                work_mem_similarity[:, -self.work_mem.get_v_size(gi) :],
                            ],
                            1,
                        ),
                        top_k=self.top_k,
                        inplace=True,
                    )
                else:
                    # no long-term memory for this group
                    affinity_one_group = do_softmax(
                        work_mem_similarity[:, -self.work_mem.get_v_size(gi) :],
                        top_k=self.top_k,
                        inplace=(gi == num_groups - 1),
                    )
                affinity.append(affinity_one_group)

        else:
            if self.enable_long_term:
                affinity, usage = do_softmax(
                    similarity,
                    inplace=(num_groups == 1),
                    top_k=self.top_k,
                    return_usage=True,
                )

            else:
                affinity = do_softmax(
                    similarity,
                    inplace=(num_groups == 1),
                    top_k=self.top_k,
                    return_usage=False,
                )

            affinity = [affinity]

            # compute affinity group by group as later groups only have a subset of keys
            for gi in range(1, num_groups):
                affinity_one_group = do_softmax(
                    similarity[:, -self.work_mem.get_v_size(gi) :],
                    top_k=self.top_k,
                    inplace=(gi == num_groups - 1),
                )
                affinity.append(affinity_one_group)

        # Shared affinity within each group
        all_readout_mem = torch.cat(
            [self._readout(affinity[gi], gv) for gi, gv in enumerate(all_memory_value)],
            0,
        )

        return all_readout_mem, usage

    def add_memory(self, key, shrinkage, value, objects, selection=None):
        # key: 1*C*H*W
        # value: 1*num_objects*C*H*W
        # objects contain a list of object indices
        if self.H is None or self.reset_config:
            self.reset_config = False
            self.H, self.W = key.shape[-2:]
            self.HW = self.H * self.W
            if self.enable_long_term:
                # convert from num. frames to num. nodes
                self.min_work_elements = self.min_mt_frames * self.HW
                self.max_work_elements = self.max_mt_frames * self.HW

        # key:   1*C*N
        # value: num_objects*C*N
        key = key.flatten(start_dim=2)
        shrinkage = shrinkage.flatten(start_dim=2)
        value = value[0].flatten(start_dim=2)

        self.CK = key.shape[1]
        self.CV = value.shape[1]

        if selection is not None:
            if not self.enable_long_term:
                warnings.warn(
                    "the selection factor is only needed in long-term mode", UserWarning
                )
            selection = selection.flatten(start_dim=2)

        self.work_mem.add(key, value, shrinkage, selection, objects)

        # long-term memory cleanup
        if self.enable_long_term:
            # Do memory compressed if needed
            if self.work_mem.size >= self.max_work_elements:
                # print('remove memory')
                # Remove obsolete features if needed
                if self.long_mem.size >= (self.max_long_elements - self.num_prototypes):
                    self.long_mem.remove_obsolete_features(
                        self.max_long_elements - self.num_prototypes
                    )

                self.compress_features()

    def create_hidden_state(self, n, sample_key):
        # n is the TOTAL number of objects
        h, w = sample_key.shape[-2:]
        if self.hidden is None:
            self.hidden = torch.zeros(
                (1, n, self.hidden_dim, h, w), device=sample_key.device
            )
        elif self.hidden.shape[1] != n:
            self.hidden = torch.cat(
                [
                    self.hidden,
                    torch.zeros(
                        (1, n - self.hidden.shape[1], self.hidden_dim, h, w),
                        device=sample_key.device,
                    ),
                ],
                1,
            )

        assert self.hidden.shape[1] == n

    def set_hidden(self, hidden):
        self.hidden = hidden

    def get_hidden(self):
        return self.hidden

    def compress_features(self):
        HW = self.HW
        candidate_value = []
        total_work_mem_size = self.work_mem.size
        for gv in self.work_mem.value:
            # Some object groups might be added later in the video
            # So not all keys have values associated with all objects
            # We need to keep track of the key->value validity
            mem_size_in_this_group = gv.shape[-1]
            if mem_size_in_this_group == total_work_mem_size:
                # full LT
                candidate_value.append(gv[:, :, HW : -self.min_work_elements + HW])
            else:
                # mem_size is smaller than total_work_mem_size, but at least HW
                assert HW <= mem_size_in_this_group < total_work_mem_size
                if mem_size_in_this_group > self.min_work_elements + HW:
                    # part of this object group still goes into LT
                    candidate_value.append(gv[:, :, HW : -self.min_work_elements + HW])
                else:
                    # this object group cannot go to the LT at all
                    candidate_value.append(None)

        # perform memory consolidation
        prototype_key, prototype_value, prototype_shrinkage = self.consolidation(
            *self.work_mem.get_all_sliced(HW, -self.min_work_elements + HW),
            candidate_value,
        )

        # remove consolidated working memory
        self.work_mem.sieve_by_range(
            HW, -self.min_work_elements + HW, min_size=self.min_work_elements + HW
        )

        # add to long-term memory
        self.long_mem.add(
            prototype_key,
            prototype_value,
            prototype_shrinkage,
            selection=None,
            objects=None,
        )
        # print(f'long memory size: {self.long_mem.size}')
        # print(f'work memory size: {self.work_mem.size}')

    def consolidation(
        self,
        candidate_key,
        candidate_shrinkage,
        candidate_selection,
        usage,
        candidate_value,
    ):
        # keys: 1*C*N
        # values: num_objects*C*N
        N = candidate_key.shape[-1]

        # find the indices with max usage
        _, max_usage_indices = torch.topk(
            usage, k=self.num_prototypes, dim=-1, sorted=True
        )
        prototype_indices = max_usage_indices.flatten()

        # Prototypes are invalid for out-of-bound groups
        validity = [
            prototype_indices >= (N - gv.shape[2]) if gv is not None else None
            for gv in candidate_value
        ]

        prototype_key = candidate_key[:, :, prototype_indices]
        prototype_selection = (
            candidate_selection[:, :, prototype_indices]
            if candidate_selection is not None
            else None
        )

        """
        Potentiation step
        """
        similarity = get_similarity(
            candidate_key, candidate_shrinkage, prototype_key, prototype_selection
        )

        # convert similarity to affinity
        # need to do it group by group since the softmax normalization would be different
        affinity = [
            (
                do_softmax(similarity[:, -gv.shape[2] :, validity[gi]])
                if gv is not None
                else None
            )
            for gi, gv in enumerate(candidate_value)
        ]

        # some values can be have all False validity. Weed them out.
        affinity = [
            aff if aff is None or aff.shape[-1] > 0 else None for aff in affinity
        ]

        # readout the values
        prototype_value = [
            self._readout(affinity[gi], gv) if affinity[gi] is not None else None
            for gi, gv in enumerate(candidate_value)
        ]

        # readout the shrinkage term
        prototype_shrinkage = (
            self._readout(affinity[0], candidate_shrinkage)
            if candidate_shrinkage is not None
            else None
        )

        return prototype_key, prototype_value, prototype_shrinkage
