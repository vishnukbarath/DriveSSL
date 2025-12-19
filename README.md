# DriveSSL 🚗🧠  
Self-Supervised and Multi-Task Learning on BDD100K

DriveSSL is a deep learning research project focused on learning robust visual representations for autonomous driving scenarios using the BDD100K dataset.

The project follows a staged learning strategy:
1. Self-supervised pretraining using SimCLR
2. Supervised multi-head classification (weather, scene, time-of-day)
3. Linear evaluation and fine-tuning
4. Detailed error analysis using confusion matrices

The goal is to analyze representation quality, task interference, and failure modes in real-world driving data.
## Repository Structure

```text
DriveSSL/
├── src/
│   ├── models/
│   │   ├── resnet_simclr.py
│   │   ├── simclr.py
│   │   ├── multihead_model.py
│   │   ├── linear_probe.py
│   ├── datasets/
│   │   ├── ssl_dataset.py
│   │   ├── bdd_multihead.py
│   │   ├── bdd_weather.py
│   │   ├── bdd_linear.py
│   ├── losses/
│   │   ├── nt_xent.py
│   │   ├── nt_xent_custom.py
│   └── utils/
│       └── device.py
├── scripts/
│   ├── train_simclr.py
│   ├── train_multihead.py
│   ├── train_confusion_multihead.py
│   ├── confusion_multihead.py
│   ├── train_linear_eval.py
│   ├── train_finetune.py
│   ├── train_weather_linear.py
│   └── visualize_embeddings.py
├── experiments/
│   ├── multihead/
│   ├── confusion/
│   ├── linear_eval/
│   └── weather/
└── README.md
