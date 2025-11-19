project<br>
│<br>
├── configs/<br>
│   ├── train_cfg.py            # MainCfg <br>
│   ├── zo_cfg.py               # MainCfg --use SZO's ZO<br>
│<br>
├── data/<br>
│   ├── nmini_dataset.py        # NMiniDataset, load_npz<br>
│   ├── dataloader.py           # make_dataloaders<br>
│<br>
├── models/<br>
│   ├── srnn.py                 # SRNN + Adapter<br>
│   ├── decoder.py              # logits readout modules<br>
│<br>
├── optim/<br>
│   ├── spsa_ops.py             # spsa_update_Wrec / Win / ce_loss_no_grad<br>
│   ├── sgd_ops.py              # 手工 SGD for W_out<br>
│<br>
├── monitors/<br>
│   ├── synchrony_monitor.py    # 同步检测<br>
│<br>
├── utils/<br>
│   ├── seed.py                 # set_seed<br>
│   ├── device.py            #auto_select_device<br>
│   ├── plot.py                 # savefig_safe, raster, loss figures<br>
│   ├── metrics.py              # evaluate, evaluate_loss<br>
│<br>
├── trainer/<br>
│   ├── trainer.py              <br>
│   ├── trainer.py              # ZO' Train<br>
│<br>
├── main.py<br
├── main_zo.py                  # use SZO's update in our framework  <br>          
├── train_cifar10_zo_subspace_rge.py    #SZO's original<br>
└── README.md<br>
