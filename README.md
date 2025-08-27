## Overview

**Cross-time self-distillation.** We add a cross-time loss to teacher-student SSL: the teacher encodes frame $t_1$, the student encodes a nearby frame $t_2$, and we align their outputs across time. We integrate this loss into image-level [iBOT](https://arxiv.org/abs/2111.07832) [1] and object-level [ODIS](https://arxiv.org/abs/2506.05409) [2] self-supervised learning frameworks. We pretrain models on the [VidOR](https://xdshang.github.io/docs/vidor.html) [3], and [WT Venice](https://huggingface.co/datasets/shawshankvkt/Walking_Tours) [4] datasets. 

---
**Clone and Install**

```bash
git clone https://github.com/KarahanS/Cross-Time-Self-Distillation.git
cd Cross-Time-Self-Distillation
pip install -r requirements.txt
```

**Run Experiments**


1. `data/`: Scripts to build subsets of the VidOR and WT Venice datasets as described in the paper. Uses [FFmpeg](https://ffmpeg.org/) to extract frames at fixed intervals.
2. `training/`: Data loaders and training loops for iBOT/ODIS with the cross-time loss, supporting both VidOR and WT Venice.
3. `jobs/`: SLURM scripts for training and evaluation, plus `.bat` files for running training locally on Windows (e.g., for debugging).
4. `.`: Top-level evaluation and utility scripts, shared helpers, `run.sh` (for cluster) and `requirements.txt`.

---

1. Jinghao Zhou, Chen Wei, Huiyu Wang, Wei Shen, Cihang Xie, Alan Yuille, and Tao Kong. iBOT: Image bert pre-training with online tokenizer, 2022. URL https://arxiv.org/abs/2111.07832.
2. Çağlar Hızlı, Çağatay Yıldız, and Pekka Marttinen. Object-level self-distillation for vision pretraining, 2025. URL https://arxiv.org/abs/2506.05409.120
3. Xindi Shang, Donglin Di, Junbin Xiao, Yu Cao, Xun Yang, and Tat-Seng Chua. Annotating objects and relations in user-generated videos. In Proceedings of the 2019 on International Conference on Multimedia Retrieval, pages 279–287. ACM, 2019.
4. Shashanka Venkataramanan, Mamshad Nayeem Rizve, João Carreira, Yuki M. Asano, and Yannis Avrithis. Is imagenet worth 1 video? learning strong image encoders from 1 long unlabelled video, 2024. URL https://arxiv.org/abs/2310.08584
