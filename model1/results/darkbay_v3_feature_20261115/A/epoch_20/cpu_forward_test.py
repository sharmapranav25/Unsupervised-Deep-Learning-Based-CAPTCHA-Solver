def forward_test(self,
                 memory: Tensor,
                 pos_query: Tensor = None,
                 max_length: Optional[int] = None) -> Tensor:
    _device = memory.device
    testing = max_length is None
    max_length = (self.max_label_length if max_length is None else min(
        max_length, self.max_label_length))
    bs = memory.shape[0]
    # +1 for <eos> at end of sequence.
    num_steps = max_length + 1
    # memory = self.encode(images)

    # Query positions up to `num_steps`
    if pos_query is None:
        pos_queries = self.pos_queries[:, :num_steps].expand(bs, -1, -1)
    else:
        pos_queries = pos_query

    # Special case for the forward permutation. Faster than using `generate_attn_masks()`
    tgt_mask = query_mask = torch.triu(
        torch.full((num_steps, num_steps), float('-inf'), device=_device),
        1)
    self.attn_maps = []
    if self.decode_ar:
        tgt_in = torch.full((bs, num_steps),
                            self.pad_id,
                            dtype=torch.long,
                            device=_device)
        tgt_in[:, 0] = self.bos_id

        logits = []
        for i in range(num_steps):
            j = i + 1  # next token index
            # Efficient decoding:
            # Input the context up to the ith token. We use only one query (at position = i) at a time.
            # This works because of the lookahead masking effect of the canonical (forward) AR context.
            # Past tokens have no access to future tokens, hence are fixed once computed.
            tgt_out = self.decode(
                tgt_in[:, :j],
                memory,
                tgt_mask[:j, :j],
                tgt_query=pos_queries[:, i:j],
                tgt_query_mask=query_mask[i:j, :j],
                pos_query=pos_queries,
            )
            self.attn_maps.append(self.decoder.layers[-1].attn_map)
            # the next token probability is in the output's ith token position
            p_i = self.head(tgt_out)
            logits.append(p_i)
            if j < num_steps:
                # greedy decode. add the next token index to the target input
                tgt_in[:, j] = p_i.squeeze().argmax(-1)
                # Efficient batch decoding: If all output words have at least one EOS token, end decoding.
                if testing and (tgt_in == self.eos_id).any(dim=-1).all():
                    break

        logits = torch.cat(logits, dim=1)
    else:
        # No prior context, so input is just <bos>. We query all positions.
        tgt_in = torch.full((bs, 1),
                            self.bos_id,
                            dtype=torch.long,
                            device=_device)
        tgt_out = self.decode(tgt_in,
                              memory,
                              tgt_query=pos_queries,
                              pos_query=pos_queries)
        logits = self.head(tgt_out)

    if self.refine_iters:
        # For iterative refinement, we always use a 'cloze' mask.
        # We can derive it from the AR forward mask by unmasking the token context to the right.
        query_mask[torch.triu(
            torch.ones(num_steps,
                       num_steps,
                       dtype=torch.bool,
                       device=_device), 2)] = 0
        bos = torch.full((bs, 1),
                         self.bos_id,
                         dtype=torch.long,
                         device=_device)
        for i in range(self.refine_iters):
            # Prior context is the previous output.
            tgt_in = torch.cat([bos, logits[:, :-1].argmax(-1)], dim=1)
            tgt_padding_mask = (tgt_in == self.eos_id).int().cumsum(
                -1) > 0  # mask tokens beyond the first EOS token.
            tgt_out = self.decode(
                tgt_in,
                memory,
                tgt_mask,
                tgt_padding_mask,
                tgt_query=pos_queries,
                tgt_query_mask=query_mask[:, :tgt_in.shape[1]],
                pos_query=pos_queries,
            )
            logits = self.head(tgt_out)

    return F.softmax(logits, -1)
