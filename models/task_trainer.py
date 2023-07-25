import os
import logging
import time
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from utils.utils import save_model, time_since, get_metrics, ContrastiveLoss
from utils.dist import is_dist_avail_and_initialized, is_main_process


logger = logging.getLogger(__name__)


class TaskTrainer:

    def __init__(self, model, output_dir, grad_norm_clip=1.0, device='cuda',
                 learning_rate=1e-4, max_epochs=10, use_amp=True, distributed=False, model_type='siamese'):
        self.model = model
        self.output_dir = output_dir
        self.grad_norm_clip = grad_norm_clip
        self.writer = SummaryWriter(self.output_dir)
        self.learning_rate = learning_rate
        self.device = device
        self.n_epochs = max_epochs
        self.use_amp = use_amp
        self.distributed = distributed
        self.model_type = model_type

        self.ctl_loss = ContrastiveLoss()
        self.bce_loss = nn.CrossEntropyLoss()
    
    def fit(self, train_loader, test_loader=None, val_loader=None, save_ckpt=True):
        model = self.model
        raw_model = model.module if hasattr(model, "module") else model
        optimizer = raw_model.configure_optimizers(self.learning_rate)
        
        if torch.cuda.is_available() and self.device == 'cuda' and self.distributed:  # for distributed parallel
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model.cuda())
            local_rank = int(os.environ['LOCAL_RANK'])
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[local_rank])
        
        def run_bert_forward(batch):
            seq, label = batch
            label = label.to(self.device)
            seq_pred, seq_embd  = model.forward(seq)
            loss = self.bce_loss(seq_pred, label)
            return loss
        
        def run_siamese_forward(batch):
            seq1, seq2, label1, label2, label = batch
            label1, label2, label = label1.to(self.device), label2.to(self.device), label.to(self.device)
            seq1_pred, seq1_embd  = model.forward(seq1)
            seq2_pred, seq2_embd  = model.forward(seq2)
            cl_loss = self.ctl_loss(seq1_embd, seq2_embd, label)

            seq1_loss = self.bce_loss(seq1_pred, label1)
            seq2_loss = self.bce_loss(seq2_pred, label2)
                            
            loss = 100*(seq1_loss + seq2_loss) + cl_loss
            # logger.info(f"epoch {epoch + 1}: seq1 loss {seq1_loss.item():.5f}, seq2 loss {seq2_loss.item():.5f}, cl loss {cl_loss.item():.5f}, total loss {loss.item():.5f}")
            return loss

        def run_epoch(split):
            is_train = split == 'train'
            model.train(is_train)
            loader = train_loader if is_train else test_loader
            if self.distributed:
                loader.sampler.set_epoch(epoch)   # for distributed parallel

            losses = []
            # pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            pbar = enumerate(loader)
            for it, batch in pbar:
                with torch.set_grad_enabled(is_train):
                    if self.device == 'cuda':
                        with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.use_amp):
                            loss = run_siamese_forward(batch) if self.model_type == 'siamese' else run_bert_forward(batch)
                            loss = loss.mean()  # collapse all losses if they are scattered on multiple gpus
                            losses.append(loss.item())
                    else:
                        loss = run_siamese_forward(batch) if self.model_type == 'siamese' else run_bert_forward(batch)
                    losses.append(loss.item())

                if is_train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_norm_clip)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            loss = float(np.mean(losses))
            logger.info(f'{split}, epoch: {epoch + 1}/{self.n_epochs}, loss: {loss:.4f}')
            self.writer.add_scalar(f'{split}_loss', loss, epoch + 1)
            return loss

        for epoch in range(self.n_epochs):
            train_loss = run_epoch('train')
            if test_loader is not None and is_main_process():
                test_loss = run_epoch('test')
            if val_loader is not None and is_main_process() and not self.distributed:
                self._eval_model(val_loader, epoch)

            curr_loss = test_loss if 'test_loss' in locals() else train_loss
            # save model in each epoch
            if self.output_dir is not None and save_ckpt:
                self._save_model(self.output_dir, str(epoch+1), curr_loss)

        if self.output_dir is not None and save_ckpt:  # save final model
            self._save_model(self.output_dir, 'final', curr_loss)

    def _save_model(self, base_dir, info, valid_loss):
        """
        Save a copy of the model with format: model_{info}_{valid_loss}
        """
        base_name = f'model_{info}_{valid_loss:.3f}'
        # logger.info(f'Save model {base_name}')
        if not is_dist_avail_and_initialized() or is_main_process():  # for distributed parallel
            save_model(self.model, base_dir, base_name)
    
    def predict(self, X_test):
        with torch.set_grad_enabled(False):
            output, _ = self.model.forward(X_test)
            y_hat = output.squeeze()
            y_hat = y_hat.argmax(axis=1)
            return y_hat

    def _eval_model(self, val_loader, epoch):
        y_test = []
        y_test_hat = []

        for x, y in val_loader:
            y_hat = self.predict(x)
            logger.debug(f'y_hat: {y_hat.shape}, y: {y.shape}')
            y_test_hat.append(y_hat.cpu().numpy())
            y_test.append(y.cpu().numpy())

        y_test = np.concatenate(y_test, axis=0)
        y_test_hat = np.concatenate(y_test_hat, axis=0)
        logger.debug(f"y_test: {y_test.shape}, y_test_hat: {y_test_hat.shape}")
        acc, sn, sp, mcc, auroc = get_metrics(y_test_hat, y_test, print_metrics=False)
        logger.info(f'eval, epoch: {epoch + 1}/{self.n_epochs}, acc: {100*acc:.2f}, sn: {100*sn:.2f}, sp: {100*sp:.2f}, mcc: {mcc:.3f}, auroc: {auroc:.3f}')
        self.writer.add_scalar('mcc', mcc, epoch + 1)
        logger.debug('eval finished')

