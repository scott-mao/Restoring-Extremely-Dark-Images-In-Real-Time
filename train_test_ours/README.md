# Training and testing
The network was trained for 1 Million iterations on RAW images. As RAW images are bulky the network was taking 7 days to train. We thus loaded the entire decoded dataset into CPU RAM which reduced the training to less than 24 hours. We thus recommend doing the same at cost of high CPU RAM utilisation (app. 60 GB).

We used `train.py` to do training and testing. Please refer the detailed comments in this file for more information.

Just for your convenience we provide files for training and testing related works. Please refer their original repositories to know more detials.
