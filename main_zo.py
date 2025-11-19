# main_zo.py
from configs.zo_cfg import ZOTrainConfig
from trainer.zo_trainer import ZOTrainer


def main():
    cfg = ZOTrainConfig()
    trainer = ZOTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
