import torch
from base.trainer import Trainer


class HANTrainer(Trainer):

    def __init__(self,
                 model,
                 optimizer,
                 criterion,
                 callbacks=None,
                 device=None):

        super(
            HANTrainer,
            self).__init__(
            model,
            optimizer,
            criterion,
            callbacks,
            device)

    def _iter(self, batch, gradient_clip, is_train=True):
        
        batch = [t.to(self.device) for t in batch]

        X, y, word_masked, sent_masked = batch

        logits = self.model(X, word_masked, sent_masked)
        loss = self.criterion(logits, y)

        if is_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), gradient_clip)
            self.optimizer.step()
            self.model.zero_grad()

        y_true_classes = y.detach().cpu().numpy().tolist()
        y_pred_classes = logits.argmax(-1).detach().cpu().numpy().tolist()

        return loss.item(), y_true_classes, y_pred_classes
