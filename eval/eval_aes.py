import os
import torch
import torch.nn as nn
import clip
from PIL import Image
import urllib.request
from tqdm import tqdm
import argparse

CLIP_MODEL_NAME = "ViT-L/14" 

PREDICTOR_URL = "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac+logos+ava1-l14-linearMSE.pth"
MODEL_FILENAME = "/root/autodl-tmp/concept-prototype/data/sac+logos+ava1-l14-linearMSE.pth"

class AestheticPredictor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available(): 
        return "mps"
    return "cpu"

def load_models(device):
    clip_model, preprocess = clip.load(CLIP_MODEL_NAME, device=device)
    
    if not os.path.exists(MODEL_FILENAME):
        try:
            urllib.request.urlretrieve(PREDICTOR_URL, MODEL_FILENAME)
        except Exception as e:
            exit()

    predictor = AestheticPredictor(768).to(device)
    
    state_dict = torch.load(MODEL_FILENAME, map_location=device)
    predictor.load_state_dict(state_dict)
    predictor.eval()
    
    return clip_model, preprocess, predictor

def score_images(folder_path, sort_output=True):
    device = get_device()
    
    clip_model, preprocess, predictor = load_models(device)
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    image_files = [f for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in image_extensions]
    
    results = []
    
    with torch.no_grad():
        for img_file in tqdm(image_files):
            try:
                img_path = os.path.join(folder_path, img_file)
                image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
                image_features = clip_model.encode_image(image)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                score = predictor(image_features.float())
                
                score_val = round(score.item(), 4)
                results.append({"file": img_file, "score": score_val})
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
                continue

    if sort_output:
        results.sort(key=lambda x: x["score"], reverse=True)
        
    return results

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True)
    parser.add_argument("--save_csv", action="store_true")
    parser.add_argument("--min_score", type=float, default=0.0)
    
    args = parser.parse_args()
    
    scores = score_images(args.folder)
    
    if scores:
        filtered_scores = [s for s in scores if s['score'] >= args.min_score]
        
        for item in filtered_scores[:10]:
            print(f"Score: {item['score']} | File: {item['file']}")
            
        mean_score = sum(item['score'] for item in filtered_scores) / len(filtered_scores) if filtered_scores else 0
        if args.save_csv:
            import csv
            csv_file = "aesthetic_scores_v2_5.csv"
            with open(csv_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["file", "score"])
                writer.writeheader()
                writer.writerows(filtered_scores)
    else:
        print("No valid images found in the specified folder.")