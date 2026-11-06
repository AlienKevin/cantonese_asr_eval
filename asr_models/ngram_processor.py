from transformers import LogitsProcessor
import kenlm
import math
import torch
from typing import Union


LOG_BASE_CHANGE_FACTOR = 1.0 / math.log10(math.e)


class NgramLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        lm_model: Union[str, "kenlm.Model"],
        lm_alpha: float = 0.5,
        eos_token_id: int = 50256,
        pad_token_id: int = 50257,
    ):
        self.lm: "kenlm.Model" = (
            kenlm.Model(lm_model) if type(lm_model) == str else lm_model
        )
        self.lm_alpha = lm_alpha
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        n_beams, n_vocab = scores.shape

        # print(input_ids.shape, scores.shape)
        # (beam_size, seq_len) (beam_size, vocab_size)

        for i in range(n_beams):
            # skip first 4 special tokens
            prefix = input_ids[i].tolist()

            if len(prefix) > 4:
                curr_state = kenlm.State()
                next_state = kenlm.State()
                self.lm.BeginSentenceWrite(curr_state)
                lm_score = []
                prob = 0.0

                for k in range(len(prefix[4:])):
                    # chr(prefix[k + 4] + 100) since we encode the text with 100 offset
                    prob += self.lm.BaseScore(
                        curr_state, chr(prefix[k + 4] + 100), next_state
                    )
                    curr_state, next_state = next_state, curr_state

                # save last state so that we do not have to recompute the whole sentence
                last_state = curr_state
                # calculate all log10 probabilities of all tokens
                # this is not efficient: https://github.com/kpu/kenlm/issues/367
                for k in range(n_vocab):
                    if k in [
                        self.eos_token_id,
                        self.pad_token_id,
                    ]:
                        lm_score.append(0)
                        continue

                    new_token_state = kenlm.State()
                    new_token_score = self.lm.BaseScore(
                        last_state, chr(k + 100), new_token_state
                    )
                    lm_score.append(new_token_score)

                lm_score = torch.FloatTensor(lm_score).to(scores.device)
                lm_score = self.lm_alpha * lm_score

                scores[i] = scores[i] + lm_score

        return scores
