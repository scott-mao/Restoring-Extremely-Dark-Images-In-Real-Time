# Training and testing
The network was trained for 1 Million iterations on RAW images. As RAW images are bulky the network was taking 7 days to train. We thus loaded the entire decoded dataset into CPU RAM which reduced the training to less than 24 hours. We thus recommend doing the same at cost of high CPU RAM utilisation (app. 60 GB).

We used `train.py` to do training and testing. Please refer the detailed comments in this file for more information. Unlike demo files, we did not rigorously test this folder after uploading to GITHUB. Small changes may be required to execute it on your system.

For your convenience we also provide files for training and testing related works. These files are only for convenience and not a replacement to original repositories.
