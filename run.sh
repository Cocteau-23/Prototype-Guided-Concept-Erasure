python src/generate_prompts.py --out data/prompts/nudity.csv --category nudity \
    --n_per_category 100 --dropout 0.2 --schema "src/schema.json"

python src/generate_samples.py --csv_path data/prompts/nudity.csv \
    --out_dir data/samples/nudity --num_images 4

python src/train_sd14.py --clip_model_path "openai/clip-vit-large-patch14" \
    --pos_image_dir data/samples/nudity/on \
    --neg_image_dir data/samples/nudity/off \
    --output_path prototypes/nudity.pt

python src/generate_sd14.py --prototype_path prototypes/nudity.pt \
    --csv_path data/concept_csv/I2P_nudity.csv \
    --out_dir output/nudity