project/
│
├── configs/
│   ├── train_cfg.py            # MainCfg 
│   ├── zo_cfg.py               # MainCfg --use SZO's ZO
│
├── data/
│   ├── nmini_dataset.py        # NMiniDataset, load_npz
│   ├── dataloader.py           # make_dataloaders
│
├── models/
│   ├── srnn.py                 # SRNN + Adapter
│   ├── decoder.py              # logits readout modules
│
├── optim/
│   ├── spsa_ops.py             # spsa_update_Wrec / Win / ce_loss_no_grad
│   ├── sgd_ops.py              # 手工 SGD for W_out
│
├── monitors/
│   ├── synchrony_monitor.py    # 同步检测
│
├── utils/
│   ├── seed.py                 # set_seed
│   ├── device.py            #auto_select_device
│   ├── plot.py                 # savefig_safe, raster, loss figures
│   ├── metrics.py              # evaluate, evaluate_loss
│
├── trainer/
│   ├── trainer.py              
│   ├── trainer.py              # ZO' Train
│
├── main.py
├── main_zo.py                  # use SZO's update in our framework            
├── requirements.txt
└── README.md
