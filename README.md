# On Adversarial Robustness of Large-scale Audio Visual Learning
repository for our ICASSP 2022 paper:http://arxiv.org/abs/2203.12122
also covers our ICASSP 2021 paper: https://arxiv.org/abs/2011.07430   (Audio-Visual Event Recognition through the lens of Adversary)

TL;DR: Watch our video at: https://www.youtube.com/watch?v=KQceFzZe7rg
Slides: https://sigport.org/sites/default/files/docs/icassp2022_slides.pdf
Brief intro to the pipeline:

You need 64x400 precomputed feature to run this pipeline, stored in .h5 file format, we are still figuring out where to host them. 
Our loader is optimized for this precomputed feature, for computing feature from .wav on the fly see our new implementation.

To train:
see the tune.sh file and pick a model you want to train

To test:
see checkPerformance-xxx.ipynb


This repo contains a lot of scrap material for our experiments, for training a model, we suggest you go to our newer version of implementation.
The newer version of implementation can be found at: https://github.com/lijuncheng16/AudioTaggingDoneRight
This repository is good for testing your pre-trained models.
