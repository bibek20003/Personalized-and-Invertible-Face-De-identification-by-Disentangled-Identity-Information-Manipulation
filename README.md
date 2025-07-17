# Personalized and Invertible Face De-identification

This project implements a reversible face de-identification pipeline using disentangled identity manipulation, inspired by the ICCV 2021 paper.

---

## 🔧 How to Use This Project

### 📁 Step 1: Prepare Your Dataset

Start by selecting a dataset of aligned face images (preferably 256×256 resolution). In our case, we use the **CelebA** dataset.

Place your dataset in a directory (e.g., `Input_Img/`) and ensure it's organized for training.

---

### 🏋️‍♂️ Step 2: Train the Model

Open `train.py` and update the dataset path to point to your input images.

Then, run:

```bash
python train.py
```

During training, the following directories will be created:

- `checkpoints/` – contains the trained model weights:
  - `attribute_encoder.pth`
  - `generator.pth`
  - `discriminator.pth`
- `outputs/` – contains visual samples of generated outputs during training

> ℹ️ Note: Training for more epochs (e.g., 50 or more) will improve image quality but take more time. You can adjust the number of epochs in `train.py`.

---

### 🔒 Step 3: Protect (De-Identify) Images

Use the `protect.py` script to generate protected versions of your images using your trained model and a password.

> ✅ Before running, make sure to update the model weight paths in `protect.py` to point to your `checkpoints/` folder.

Run this command:

```bash
python protect.py \
  --input_dir Input_Img/fake_class \
  --output_dir protected \
  --password 123456 \
  --privacy 7
```

- `--input_dir`: Path to the folder containing original images
- `--output_dir`: Where to save the protected images
- `--password`: Password to control de-identification (must match during recovery)
- `--privacy`: Privacy level (typically between 5 and 9)

---

### 🔓 Step 4: Recover the Original Image

Use `recover.py` to reverse the de-identification and recover the original face.

> ✅ Make sure to update model weight paths in `recover.py` as well.

Run the following:

```bash
python recover.py \
  --image protected/your_image.jpg \
  --password 123456 \
  --privacy 7
```

- `--image`: Path to the protected image you want to recover
- The password and privacy level **must match** those used during protection.

---

By following these steps, you can train your own reversible face de-identification system, protect faces using identity rotation, and recover them securely with the correct password and settings.





### Thank You 
