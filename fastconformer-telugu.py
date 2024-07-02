
import nemo.collections.asr as nemo_asr
import copy
from omegaconf import OmegaConf, open_dict
import torch
import pytorch_lightning as ptl

model = nemo_asr.models.ASRModel.from_pretrained("nvidia/stt_en_fastconformer_ctc_large")
model.change_vocabulary(new_tokenizer_dir='resources/tokenizer_spe_bpe_v1024', new_tokenizer_type='bpe')

cfg = copy.deepcopy(model.cfg)

with open_dict(cfg):
    cfg.train_ds.manifest_filepath = '/home/ubuntu/users/bharath/resources/final_te_manifests/train_manifest.json'
    cfg.test_ds.manifest_filepath = 'resources/final_te_manifests/test_manifest.json'
    cfg.validation_ds.manifest_filepath = 'resources/final_te_manifests/dev_manifest.json'
    cfg.optim.lr = 0.0005
    cfg.optim.sched.min_lr = 0.000001
    cfg.batch_size=19
    cfg.train_ds.batch_size=19
    cfg.test_ds.batch_size=19
    cfg.validation_ds.batch_size=19

model.setup_training_data(cfg.train_ds)
model.setup_multiple_validation_data(cfg.validation_ds)
model.setup_multiple_test_data(cfg.test_ds)

model.cfg = cfg

EPOCHS = 1 

torch.set_float32_matmul_precision('medium')
torch.cuda.is_available()

trainer = ptl.Trainer(devices=4,
                      accelerator='gpu',
                      max_epochs=EPOCHS,
                      enable_checkpointing=True,
                      logger=True,
                      log_every_n_steps=50,
                      check_val_every_n_epoch=1,
                      strategy = 'ddp' 
                      )


# Setup model with the trainer
model.set_trainer(trainer)

# Finally, update the model's internal config
model.cfg = model._cfg

trainer.fit(model)

model.save_to("resources/fastconformer-telugu.nemo")