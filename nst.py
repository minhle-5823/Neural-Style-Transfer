import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as T
import torchvision.models as models
import torchvision.utils as vutils
import os

#  Mandatory Config based on the problem statement 
IMSIZE = 512
STEPS  = 300
ALPHA  = 1.0        # content weight
BETA   = 10000.0    # style weight
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#  Loader / Unloader 
loader = T.Compose([
    T.Resize((IMSIZE, IMSIZE)),
    T.ToTensor()
])

unloader = T.ToPILImage()

def image_loader(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image {image_path} not found!")
    image = Image.open(image_path).convert("RGB")
    # Add batch dimension and move to device
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

def tensor_to_image(tensor):
    # Clone and squeeze to remove batch dimension
    image = tensor.cpu().clone().squeeze(0)
    # Clamp pixel values to [0, 1]
    image = torch.clamp(image, 0, 1)
    return unloader(image)

#  Gram Matrix 
def gram_matrix(feat: torch.Tensor):
    # feat: (N, C, H, W)
    n, c, h, w = feat.size()
    f = feat.view(n * c, h * w)
    # Normalization as per original paper: Divide by N*C*H*W
    G = torch.mm(f, f.t())
    return G.div(n * c * h * w)

#  Loss modules 
class ContentLoss(nn.Module):
    def __init__(self, target):
        super().__init__()
        # Detach target from computation graph
        self.target = target.detach()
        # Ensure loss is on the same device
        self.loss = torch.tensor(0., device=target.device) 

    def forward(self, x):
        self.loss = nn.functional.mse_loss(x, self.target)
        return x

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super().__init__()
        # Detach target Gram Matrix
        self.target = gram_matrix(target_feature).detach()
        # Ensure loss is on the same device
        self.loss = torch.tensor(0., device=target_feature.device) 

    def forward(self, x):
        G = gram_matrix(x)
        self.loss = nn.functional.mse_loss(G, self.target)
        return x

#  Normalization layer (ImageNet standard) 
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # broadcast with the images
        self.register_buffer('mean', mean.view(-1,1,1))
        self.register_buffer('std',  std.view(-1,1,1))
    def forward(self, img):
        return (img - self.mean) / self.std

#  Build model according to required layers 
def build_model_and_losses(cnn, norm_mean, norm_std, style_img, content_img):
    cnn = cnn.eval()
    normalization = Normalization(norm_mean, norm_std).to(device)

    content_layers = ['conv4_2']
    style_layers   = ['conv1_1','conv2_1','conv3_1','conv4_1','conv5_1']

    content_losses = []
    style_losses   = []

    model = nn.Sequential(normalization)

    conv_counters = {1:0,2:0,3:0,4:0,5:0}
    block = 1

    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            conv_counters[block] += 1
            name = f'conv{block}_{conv_counters[block]}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu{block}_{conv_counters[block]}'
            # Must create nn.ReLU(inplace=False) to allow inserting loss modules
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool{block}'
            block += 1
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn{block}_{conv_counters[block]}'
        else:
            name = layer.__class__.__name__.lower()

        model.add_module(name, layer)

        if name in content_layers:
            # Calculate target feature map for Content Loss
            target = model(content_img).detach()
            cl = ContentLoss(target)
            model.add_module(f'content_loss_{name}', cl)
            content_losses.append(cl)

        if name in style_layers:
            # Calculate target feature map for Style Loss
            target = model(style_img).detach()
            sl = StyleLoss(target)
            model.add_module(f'style_loss_{name}', sl)
            style_losses.append(sl)

    # Trim the model right after the last loss to save compute
    for i in range(len(model)-1, -1, -1):
        if isinstance(model[i], (ContentLoss, StyleLoss)):
            model = model[:i+1]
            break

    return model, style_losses, content_losses

#  Core NST (L-BFGS) 
def run_style_transfer(content_path, style_path, output_path, steps=STEPS, alpha=ALPHA, beta=BETA, init_mode="blend"):

    # Load images
    try:
        content_img = image_loader(content_path)
        style_img = image_loader(style_path)
    except FileNotFoundError as e:
        print(e)
        return

    #  Input image initialization 
    # Blend: 60% content + 40% random noise
    input_img = content_img * 0.6 + torch.randn_like(content_img) * 0.4
    input_img = input_img.to(device).detach().requires_grad_(True)
    
    #  VGG19 + normalization stats 
    cnn = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.to(device).eval()
    norm_mean = torch.tensor([0.485, 0.456, 0.406], device=device)
    norm_std  = torch.tensor([0.229, 0.224, 0.225], device=device)

    # Build model with loss modules
    model, style_losses, content_losses = build_model_and_losses(
        cnn, norm_mean, norm_std, style_img, content_img
    )

    # Use L-BFGS optimizer for higher quality result (as per original paper)
    optimizer = optim.LBFGS([input_img])

    run = [0]
    def closure():
        with torch.no_grad():
            # Clamp image to maintain valid pixel values (0-1)
            input_img.clamp_(0, 1)

        optimizer.zero_grad()
        model(input_img)

        style_score   = sum(sl.loss for sl in style_losses)
        content_score = sum(cl.loss for cl in content_losses)

        # CALCULATE TOTAL LOSS (with applied weights)
        loss = alpha * content_score + beta * style_score
        loss.backward()

        run[0] += 1
        if run[0] % 50 == 0:
            print(f"Step {run[0]:4d}/{steps:4d} | "
                      f"Content Loss: {content_score.item():.4f} | "
                      f"Style Loss: {style_score.item():.4f} | "
                      f"Total Loss: {loss.item():.4f}")

        return loss

    print(f"\nStarting style transfer: {content_path}")
    # Optimization loop
    while run[0] <= steps:
        optimizer.step(closure)

    with torch.no_grad():
        input_img.clamp_(0, 1)

    # Save result
    output_image = tensor_to_image(input_img)
    output_image.save(output_path)
    print(f"Saved result: {output_path}")

    return input_img

#  Main Execution 
if __name__ == "__main__":
    # Create directories if they don't exist
    os.makedirs("images", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    # File paths
    style_path = "images/style.jpg"
    content_paths = [
        "images/content_1.jpg",
        "images/content_2.jpg",
        "images/content_3.jpg",
        "images/content_4.jpg",
        "images/content_5.jpg"
    ]

    print("=" * 50)
    print("Neural Style Transfer - Gatys et al. Style")
    # Print config for easy tracking
    print(f"α = {ALPHA}, β = {BETA}, steps = {STEPS}, size = {IMSIZE}")
    print("=" * 50)

    # Process all content images
    for i, content_path in enumerate(content_paths, 1):
        # Check for file existence before running
        if os.path.exists(content_path) and os.path.exists(style_path):
            output_path = f"output/out_{i}.png"
            try:
                run_style_transfer(content_path, style_path, output_path, init_mode="blend")
            except Exception as e:
                print(f"Error processing {content_path}: {e}")
        else:
            # Skip if file is missing
            missing_file = content_path if not os.path.exists(content_path) else style_path
            print(f"Warning: Missing file, skipping: {missing_file}")

    print("=" * 50)
    print("Style transfer completed!")
    print("Results saved in 'output/' directory")