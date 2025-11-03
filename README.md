## 🎨 Neural Style Transfer (NST) - Artistic Style Recreation

This project implements the **Neural Style Transfer** (NST) algorithm based on the original research by Gatys et al., allowing the **content** of one image to be combined with the **artistic style** of another image.

### 📁 Directory Structure

The project has the following directory structure:

```
.
├── nst.py            # Main Python script performing NST
├── images/           # Directory for input images (INPUT)
│   ├── style.jpg     # Image containing the artistic style (e.g., "Starry Night")
│   ├── content_1.jpg # Image containing the content (natural scenery)
│   ├── content_2.jpg # Image containing the content (human portrait)
│   ├── content_3.jpg # Image containing the content (building)
│   ├── content_4.jpg # Image containing the content (animal)
│   └── content_5.jpg # Image containing the content (city ​​night scene)
└── output/           # Directory for result images (OUTPUT)
    ├── out_1.png     # Result image after style transfer
    ├── out_2.png
    ...
```

-----

### 🚀 How It Works

The `nst.py` script uses a pre-trained **VGG19** model to extract image features.

1.  **Content Loss**: Keeps the high-level features (shapes, objects) of the resulting image close to the content image at the **`conv4_2`** layer.
2.  **Style Loss**: Keeps the **Gram Matrix** (describing texture, color correlation) of the resulting image close to the style image at layers **`conv1_1`, `conv2_1`, `conv3_1`, `conv4_1`, `conv5_1`**.
3.  The **L-BFGS** optimization algorithm is used to adjust the input image (initialized from the content image) to minimize the **total loss** (Content Loss + Style Loss).

-----

### ⚙️ Main Configuration and Hyperparameter Tuning

You can **adjust the hyperparameters** at the beginning of the `nst.py` file to control the quality, speed, and artistic outcome of the style transfer.

| Parameter | Description | Default Value | Tuning Guide |
| :--- | :--- | :--- | :--- |
| **`IMSIZE`** | Input image resolution (pixels) | `512` | Increase for better detail, decrease to reduce VRAM usage and speed up. |
| **`STEPS`** | Number of optimization steps | `300` | Increase (e.g., to 500-1000) for a smoother, more refined result. |
| **`ALPHA`** | **Content Weight** | `1.0` | **α**. Controls how much to preserve the original image structure. |
| **`BETA`** | **Style Weight** | `10000.0` | **β**. Controls the intensity of the applied style/texture. |

**Crucial Tuning Point:** The **ratio** of **`BETA` / `ALPHA`** is the most critical factor.

  * **Increase `BETA` / `ALPHA`** (e.g., $\beta=100000, \alpha=1$): Stronger style, content structure may be lost.
  * **Decrease `BETA` / `ALPHA`** (e.g., $\beta=1000, \alpha=1$): Stronger content preservation, weaker style effect.

-----

### 🖼️ Changing Input Images

You are welcome to change the style and content images to create new artistic works:

1.  **To change the Content Images:**

      * Replace or add new images (e.g., `content_6.jpg`) to the `images/` directory.
      * You will need to update the `content_paths` list within the `if __name__ == "__main__":` block of the `nst.py` file.

2.  **To change the Style Image:**

      * Rename your new style file (e.g., `new_painting.jpg`) to **`style.jpg`** and place it in the `images/` directory.
      * *Alternatively*, change the `style_path` variable in `nst.py` to point to your new style image file.

-----

### 💻 Requirements & Installation

This project requires PyTorch and torchvision libraries:

```bash
# Install PyTorch (e.g., for CUDA)
# Refer to the official PyTorch site for the correct version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Other libraries
pip install Pillow
```

### 🏃 Running the Program

```bash
python nst.py
```
