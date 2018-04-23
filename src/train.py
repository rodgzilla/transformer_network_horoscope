import time

def run_epoch(data_iter, model, loss_compute):
    """
    Standard Training and Logging Function.
    """
    start        = time.time()
    total_tokens = 0
    total_loss   = 0
    tokens       = 0

    for i, batch in enumerate(data_iter):
        out = model.forward(
            batch.src,
            batch.trg,
            batch.src_mask,
            batch.trg_mask
        )
        loss          = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss   += loss
        total_tokens += batch.ntokens
        tokens       += batch.ntokens

        if i % 50 == 1:
            elapsed = time.time() - start
            print(f'Epoch Step: {i} '
                  f'Loss: {loss / batch.ntokens:.2f} '
                  f'Tokens per Sec: {tokens / elapsed:.2f}')
            start  = time.time()
            tokens = 0

    return total_loss / total_tokens

global max_src_in_batch, max_tgt_in_batch
def batch_size_fn(new, count, sofar):
    """
    Keep augmenting batch and calculate total number of tokens + padding.
    """
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0

    max_src_in_batch = max(max_src_in_batch, len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)
    src_elements     = count * max_src_in_batch
    tgt_elements     = count * max_tgt_in_batch

    return max(src_elements, tgt_elements)

class SimpleLossCompute():
    """
    A simple loss compute and train function.
    """
    def __init__(self, generator, criterion, opt = None):
        self.generator = generator
        self.criterion = criterion
        self.opt       = opt

    def __call__(self, x, y, norm):
        x    = self.generator(x)
        loss = self.criterion(
            x.contiguous().view(
                -1,
                x.size(-1)
            ),
            y.contiguous().view(-1)
        ) / norm
        loss.backward()

        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()

        return loss.data[0] * norm
