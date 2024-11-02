# KLA-Denoising-Task
This Repository contains the files for Image Restoration from Noisy and Blur Images using Deep Learning Models

## Models Used :
- **Denoising Task**: RIDNet
- **Defect Mask Segmentation**: UNet

![LAB](https://github.com/user-attachments/assets/db4dc459-9fce-4aa1-a0bc-45fc8d7a1cd2)

## Model & Dataset Link :
- **Dataset**: https://drive.google.com/file/d/1iTM57U43L0ANn1u_bRv4dEEERQemuOXc/view?usp=drive_link
- **RIDNet**: https://drive.google.com/file/d/1cuyH0LxOmmW3_yZsL0GcsdodcwVhfcMy/view?usp=sharing
- **UNet**: https://drive.google.com/file/d/1pnXopSsV8yAq6GXf0AEsDQII2UdlVU2k/view?usp=sharing

## Steps to Organize the Dataset :
- Download the dataset from the link given above
- Use the **files.py** to organize the dataset into a new folder, **dataset** with subfolders **data**, **label** and **mask**, where data folder contains the noisy image, label folder contains the clean ground truth image, and mask folder contains the ground truth defect mask

  ![Screenshot from 2024-11-02 23-05-51](https://github.com/user-attachments/assets/47217ed1-eeac-44e9-8a35-d234a092dd43)

- Do change the class name in **Line 59** of **files.py** each time to append the images to dataset folder

## Steps to Run the Model :
- Download the required python librariies using requirements.txt
  ```
  pip install -r requirements.txt
  ```
- Clone the Repository
  ```
  git clone https://github.com/gokulmk-12/KLA-Denoising-DLI.git
  ```
- Download the model weights from the above link and paste in a new folder "saved_models" inside the clones folder
- Below is the expected contents of the cloned folder
  
  ![Screenshot from 2024-11-02 23-28-13](https://github.com/user-attachments/assets/171a6017-2396-47db-af62-6eedddbcd62a)
  
- Run main.py by using the following command. It opens a GUI in streamlit
  ```
  streamlit run main.py
  ```
- Below is a video demonstration on how to use the GUI

  https://github.com/user-attachments/assets/8f83d11a-628a-48ad-ae1d-f270e201a590

## Steps to Modify the Model :
- The main model files used for training are in the **models** folder
- The user are encouraged to change **config.py** with thier trained weights, provided they plan to use the same architecture as in **networks.py**
