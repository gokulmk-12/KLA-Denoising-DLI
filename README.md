# KLA-Denoising-Task
This Repository contains the files for Image Restoration from Noisy and Blur Images using Deep Learning Models

## Models Used :
- **Denoising Task**: RIDNet
- **Defect Mask Segmentation**: UNet

![LAB](https://github.com/user-attachments/assets/db4dc459-9fce-4aa1-a0bc-45fc8d7a1cd2)

## Dataset Link :
- **RIDNet**: https://drive.google.com/file/d/13yEwYvHD1QynOthkDTYoGJGo2RiI_dG7/view?usp=sharing
- **UNet**: https://drive.google.com/file/d/1pnXopSsV8yAq6GXf0AEsDQII2UdlVU2k/view?usp=sharing

## Steps to Run the Model :
- Clone the Repository
  ```
  git clone https://github.com/gokulmk-12/KLA-Denoising-DLI.git
  ```
- Download the model weights from the above link and paste in a new folder "saved_models" inside the clones folder
- Run main.py by using the following command. It opens a GUI in streamlit
  ```
  streamlit run main.py
  ```
- Below is a video demonstration on how to use the GUI
- The main model files used for training are in the **models** folder
- The user are encouraged to change **config.py** with thier trained weights, provided they plan to use the same architecture as in **networks.py**
