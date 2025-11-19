# main.py
from configs.train_cfg import TrainConfig
from trainer.trainer import Trainer

def main():
    cfg = TrainConfig()
    trainer = Trainer(cfg)
    trainer.train()

if __name__ == "__main__":
    main()
