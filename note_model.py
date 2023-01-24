import einops
import torch


def token_loss(target_indices,logits,mask,channel_vocab_sizes=[{"null",2},{"pitch",36},{"onset_time",32},{"duration",32}]):
    # target: (batch token channel_dist), logits: (batch token channel_dist), mask: (batch token channel)
    loss=0

    expanded_mask = []
    for ch_idx, channel_vocab_size in channel_vocab_sizes:
        expanded_mask.append(mask[:,:,ch_idx].unsqueeze(-1).expand(-1,-1,channel_vocab_size))
    expanded_mask = torch.cat(expanded_mask,dim=-1)

    for batch in range(target_indices.shape[0]):
        # filter predicted tokens.
        # filter away non masked tokens and maksked tokens in null tokens

        indices=[]
        for index in target_indices.shape[1]:
            # filter away tokens that are not masked
            if (mask[batch,predicted_token_index] == 0).all():
                    continue
            # filter away tokens where null is true and unmasked
            if (target_indices[batch,predicted_token_index,0] == 0) and (mask[batch,predicted_token_index,0] == 0):
                continue
            indices.append(index)

        for predicted_token_index in indices:
            # check if there is a mask
                for target_token_index in indices:
                    # check if target is a candidate
                    # if unmasked channels are equal then it is a candidate
                    candidate_list = []
                    if (mask[batch,target_token_index] == mask[batch,predicted_token_index]).all():
                        if((target_indices[batch,target_token_index]+1)*(1-mask) == (target_indices[batch,predicted_token_index]+1)*(1-mask)).all():
                            candidate_list.append(target_token_index)
            
                    for channel_idx, channel_vocab_size in channel_vocab_sizes:
                        if mask[batch,target_token_index,channel_idx] == 1:
                            one_hots=[]
                            for candidate in candidate_list:
                                # check if target is a match
                                # if target is a match then add to candidate list
                                one_hot = torch.zeros(channel_vocab_size, device=target_indices.device)
                                one_hot[target_indices[batch,target_token_index,channel_idx]] = 1
                                one_hots.append(one_hot)

                            one_hot_mean = torch.mean(torch.stack(one_hots),dim=0)
                            loss += torch.nn.functional.cross_entropy(logits[batch,predicted_token_index,channel_idx],one_hot_mean)
    return loss
    
    # if target is a match then add to candidate list
                        


                    
