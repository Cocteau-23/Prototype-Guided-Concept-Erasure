import argparse
import os
import torch
import lpips
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

def load_image(path):
    img = Image.open(path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    
    img_tensor = transform(img)
    return img_tensor.unsqueeze(0)

def calculate_lpips(dir0, dir1, model_type='alex', use_gpu=True):
    if not os.path.exists(dir0) or not os.path.exists(dir1):
        return

    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    loss_fn = lpips.LPIPS(net=model_type).to(device)

    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    files0 = sorted([f for f in os.listdir(dir0) if os.path.splitext(f)[1].lower() in valid_exts])
    
    dists = []
    count = 0

    for file in tqdm(files0):
        path0 = os.path.join(dir0, file)
        path1 = os.path.join(dir1, file)

        if os.path.exists(path1):
            img0 = load_image(path0).to(device)
            img1 = load_image(path1).to(device)
            
            if img0.shape != img1.shape:
                img1 = torch.nn.functional.interpolate(img1, size=(img0.shape[2], img0.shape[3]), mode='area')

            with torch.no_grad():
                dist = loss_fn(img0, img1)
            
            dists.append(dist.item())
            count += 1

        else:
            pass

    if count > 0:
        avg_dist = sum(dists) / count
        print("\n" + "="*40)
        print(f"number of images: {count}")
        print(f"mean LPIPS: {avg_dist:.6f}")
        print("="*40)
    else:
        print("missing: no valid image pairs found.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir0', type=str, required=True, help='First directory path (usually real images)')
    parser.add_argument('--dir1', type=str, required=True, help='Second directory path (usually generated images)')
    parser.add_argument('--net', type=str, default='alex', choices=['alex', 'vgg', 'squeeze'], help='Backbone network type (default: alex)')
    parser.add_argument('--gpu', action='store_true', help='Use GPU acceleration')

    args = parser.parse_args()
    
    calculate_lpips(args.dir0, args.dir1, args.net, args.gpu)