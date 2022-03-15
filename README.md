# Feedforward-NNs


(sdc-c1-gpu-augment) root@90275db275b9:/home/workspace# python training.py --imdir GTSRB/Final_Training/Images/ --epochs 30
2022-03-15 03:47:54.804916: W tensorflow/stream_executor/platform/default/dso_loader.cc:59] Could not load dynamic library 'libcudart.so.10.1'; dlerror: libcudart.so.10.1: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /opt/ros/kinetic/lib:/opt/ros/kinetic/lib/x86_64-linux-gnu:/usr/local/cuda-9.0/lib64:/usr/lib/x86_64-linux-gnu:/usr/lib:/usr/local/lib/x86_64-linux-gnu:/usr/local/lib/i386-linux-gnu:/usr/lib/x86_64-linux-gnu:/usr/lib/i386-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
2022-03-15 03:47:54.804986: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
2022-03-15 03:47:57,116 INFO     Training for 30 epochs using GTSRB/Final_Training/Images/ data
Found 4300 files belonging to 43 classes.
Using 3440 files for training.
2022-03-15 03:47:57.546216: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcuda.so.1
2022-03-15 03:47:57.571859: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2022-03-15 03:47:57.572823: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1716] Found device 0 with properties: 
pciBusID: 0000:00:04.0 name: Tesla K80 computeCapability: 3.7
coreClock: 0.8235GHz coreCount: 13 deviceMemorySize: 11.17GiB deviceMemoryBandwidth: 223.96GiB/s
2022-03-15 03:47:57.573107: W tensorflow/stream_executor/platform/default/dso_loader.cc:59] Could not load dynamic library 'libcudart.so.10.1'; dlerror: libcudart.so.10.1: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /opt/ros/kinetic/lib:/opt/ros/kinetic/lib/x86_64-linux-gnu:/usr/local/cuda-9.0/lib64:/usr/lib/x86_64-linux-gnu:/usr/lib:/usr/local/lib/x86_64-linux-gnu:/usr/local/lib/i386-linux-gnu:/usr/lib/x86_64-linux-gnu:/usr/lib/i386-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
2022-03-15 03:47:57.573324: W tensorflow/stream_executor/platform/default/dso_loader.cc:59] Could not load dynamic library 'libcublas.so.10'; dlerror: libcublas.so.10: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /opt/ros/kinetic/lib:/opt/ros/kinetic/lib/x86_64-linux-gnu:/usr/local/cuda-9.0/lib64:/usr/lib/x86_64-linux-gnu:/usr/lib:/usr/local/lib/x86_64-linux-gnu:/usr/local/lib/i386-linux-gnu:/usr/lib/x86_64-linux-gnu:/usr/lib/i386-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
2022-03-15 03:47:57.573561: W tensorflow/stream_executor/platform/default/dso_loader.cc:59] Could not load dynamic library 'libcufft.so.10'; dlerror: libcufft.so.10: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /opt/ros/kinetic/lib:/opt/ros/kinetic/lib/x86_64-linux-gnu:/usr/local/cuda-9.0/lib64:/usr/lib/x86_64-linux-gnu:/usr/lib:/usr/local/lib/x86_64-linux-gnu:/usr/local/lib/i386-linux-gnu:/usr/lib/x86_64-linux-gnu:/usr/lib/i386-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
2022-03-15 03:47:57.573794: W tensorflow/stream_executor/platform/default/dso_loader.cc:59] Could not load dynamic library 'libcurand.so.10'; dlerror: libcurand.so.10: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /opt/ros/kinetic/lib:/opt/ros/kinetic/lib/x86_64-linux-gnu:/usr/local/cuda-9.0/lib64:/usr/lib/x86_64-linux-gnu:/usr/lib:/usr/local/lib/x86_64-linux-gnu:/usr/local/lib/i386-linux-gnu:/usr/lib/x86_64-linux-gnu:/usr/lib/i386-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
2022-03-15 03:47:57.574098: W tensorflow/stream_executor/platform/default/dso_loader.cc:59] Could not load dynamic library 'libcusolver.so.10'; dlerror: libcusolver.so.10: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /opt/ros/kinetic/lib:/opt/ros/kinetic/lib/x86_64-linux-gnu:/usr/local/cuda-9.0/lib64:/usr/lib/x86_64-linux-gnu:/usr/lib:/usr/local/lib/x86_64-linux-gnu:/usr/local/lib/i386-linux-gnu:/usr/lib/x86_64-linux-gnu:/usr/lib/i386-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
2022-03-15 03:47:57.574313: W tensorflow/stream_executor/platform/default/dso_loader.cc:59] Could not load dynamic library 'libcusparse.so.10'; dlerror: libcusparse.so.10: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /opt/ros/kinetic/lib:/opt/ros/kinetic/lib/x86_64-linux-gnu:/usr/local/cuda-9.0/lib64:/usr/lib/x86_64-linux-gnu:/usr/lib:/usr/local/lib/x86_64-linux-gnu:/usr/local/lib/i386-linux-gnu:/usr/lib/x86_64-linux-gnu:/usr/lib/i386-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
2022-03-15 03:47:57.574582: W tensorflow/stream_executor/platform/default/dso_loader.cc:59] Could not load dynamic library 'libcudnn.so.7'; dlerror: libcudnn.so.7: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /opt/ros/kinetic/lib:/opt/ros/kinetic/lib/x86_64-linux-gnu:/usr/local/cuda-9.0/lib64:/usr/lib/x86_64-linux-gnu:/usr/lib:/usr/local/lib/x86_64-linux-gnu:/usr/local/lib/i386-linux-gnu:/usr/lib/x86_64-linux-gnu:/usr/lib/i386-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
2022-03-15 03:47:57.574645: W tensorflow/core/common_runtime/gpu/gpu_device.cc:1753] Cannot dlopen some GPU libraries. Please make sure the missing libraries mentioned above are installed properly if you would like to use GPU. Follow the guide at https://www.tensorflow.org/install/gpu for how to download and setup the required libraries for your platform.
Skipping registering GPU devices...
2022-03-15 03:47:57.575350: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN)to use the following CPU instructions in performance-critical operations:  AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2022-03-15 03:47:57.584298: I tensorflow/core/platform/profile_utils/cpu_utils.cc:104] CPU Frequency: 2299995000 Hz
2022-03-15 03:47:57.584666: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55a742b347b0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2022-03-15 03:47:57.584716: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
2022-03-15 03:47:57.586861: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1257] Device interconnect StreamExecutor with strength 1 edge matrix:
2022-03-15 03:47:57.586899: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1263]      
Found 4300 files belonging to 43 classes.
Using 860 files for validation.
Epoch 1/30
14/14 [==============================] - 1s 65ms/step - loss: 3.7078 - accuracy: 0.0686 - val_loss: 3.4330 - val_accuracy: 0.1279
Epoch 2/30
14/14 [==============================] - 1s 54ms/step - loss: 3.2781 - accuracy: 0.1866 - val_loss: 3.1487 - val_accuracy: 0.2070
Epoch 3/30
14/14 [==============================] - 1s 56ms/step - loss: 2.9548 - accuracy: 0.2776 - val_loss: 2.8556 - val_accuracy: 0.3302
Epoch 4/30
14/14 [==============================] - 1s 55ms/step - loss: 2.6684 - accuracy: 0.3770 - val_loss: 2.6060 - val_accuracy: 0.3523
Epoch 5/30
14/14 [==============================] - 1s 54ms/step - loss: 2.4454 - accuracy: 0.4427 - val_loss: 2.3832 - val_accuracy: 0.4640
Epoch 6/30
14/14 [==============================] - 1s 54ms/step - loss: 2.2376 - accuracy: 0.5003 - val_loss: 2.2307 - val_accuracy: 0.4988
Epoch 7/30
14/14 [==============================] - 1s 55ms/step - loss: 2.0944 - accuracy: 0.5378 - val_loss: 2.1141 - val_accuracy: 0.5291
Epoch 8/30
14/14 [==============================] - 1s 53ms/step - loss: 1.9549 - accuracy: 0.5837 - val_loss: 1.9673 - val_accuracy: 0.5651
Epoch 9/30
14/14 [==============================] - 1s 56ms/step - loss: 1.8288 - accuracy: 0.6230 - val_loss: 1.8590 - val_accuracy: 0.5733
Epoch 10/30
14/14 [==============================] - 1s 51ms/step - loss: 1.7488 - accuracy: 0.6192 - val_loss: 1.7779 - val_accuracy: 0.5826
Epoch 11/30
14/14 [==============================] - 1s 51ms/step - loss: 1.6357 - accuracy: 0.6616 - val_loss: 1.7011 - val_accuracy: 0.6279
Epoch 12/30
14/14 [==============================] - 1s 56ms/step - loss: 1.5410 - accuracy: 0.6884 - val_loss: 1.6121 - val_accuracy: 0.6453
Epoch 13/30
14/14 [==============================] - 1s 54ms/step - loss: 1.4672 - accuracy: 0.7227 - val_loss: 1.5568 - val_accuracy: 0.6512
Epoch 14/30
14/14 [==============================] - 1s 54ms/step - loss: 1.4035 - accuracy: 0.7183 - val_loss: 1.5523 - val_accuracy: 0.6581
Epoch 15/30
14/14 [==============================] - 1s 56ms/step - loss: 1.3483 - accuracy: 0.7424 - val_loss: 1.4528 - val_accuracy: 0.6826
Epoch 16/30
14/14 [==============================] - 1s 51ms/step - loss: 1.2838 - accuracy: 0.7541 - val_loss: 1.4354 - val_accuracy: 0.6802
Epoch 17/30
14/14 [==============================] - 1s 55ms/step - loss: 1.2279 - accuracy: 0.7683 - val_loss: 1.3765 - val_accuracy: 0.6907
Epoch 18/30
14/14 [==============================] - 1s 55ms/step - loss: 1.1747 - accuracy: 0.7846 - val_loss: 1.3389 - val_accuracy: 0.7093
Epoch 19/30
14/14 [==============================] - 1s 54ms/step - loss: 1.1319 - accuracy: 0.7901 - val_loss: 1.2580 - val_accuracy: 0.7384
Epoch 20/30
14/14 [==============================] - 1s 56ms/step - loss: 1.0819 - accuracy: 0.8125 - val_loss: 1.2338 - val_accuracy: 0.7500
Epoch 21/30
14/14 [==============================] - 1s 54ms/step - loss: 1.0509 - accuracy: 0.8183 - val_loss: 1.2180 - val_accuracy: 0.7314
Epoch 22/30
14/14 [==============================] - 1s 55ms/step - loss: 1.0035 - accuracy: 0.8221 - val_loss: 1.1736 - val_accuracy: 0.7488
Epoch 23/30
14/14 [==============================] - 1s 58ms/step - loss: 0.9746 - accuracy: 0.8273 - val_loss: 1.1444 - val_accuracy: 0.7465
Epoch 24/30
14/14 [==============================] - 1s 53ms/step - loss: 0.9521 - accuracy: 0.8328 - val_loss: 1.1255 - val_accuracy: 0.7395
Epoch 25/30
14/14 [==============================] - 1s 51ms/step - loss: 0.9107 - accuracy: 0.8465 - val_loss: 1.0675 - val_accuracy: 0.7698
Epoch 26/30
14/14 [==============================] - 1s 54ms/step - loss: 0.8749 - accuracy: 0.8576 - val_loss: 1.0537 - val_accuracy: 0.7756
Epoch 27/30
14/14 [==============================] - 1s 56ms/step - loss: 0.8448 - accuracy: 0.8637 - val_loss: 1.0376 - val_accuracy: 0.7849
Epoch 28/30
14/14 [==============================] - 1s 55ms/step - loss: 0.8247 - accuracy: 0.8605 - val_loss: 0.9874 - val_accuracy: 0.7930
Epoch 29/30
14/14 [==============================] - 1s 57ms/step - loss: 0.7975 - accuracy: 0.8735 - val_loss: 0.9968 - val_accuracy: 0.7895
Epoch 30/30
14/14 [==============================] - 1s 52ms/step - loss: 0.7831 - accuracy: 0.8674 - val_loss: 0.9714 - val_accuracy: 0.7814
