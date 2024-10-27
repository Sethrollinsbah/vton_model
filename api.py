from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from PIL import Image
import torch
import torchvision.transforms as transforms

# ... (rest of the imports)

app = FastAPI()

@app.post("/infer")
async def infer(image: UploadFile = File(...), clothes: UploadFile = File(...), edge: UploadFile = File(...)):
    # Load and preprocess images
    image = Image.open(image.file)
    clothes = Image.open(clothes.file)
    edge = Image.open(edge.file)

    # Convert images to tensors (adjust as needed)
    transform = transforms.Compose([
        transforms.Resize((256, 192)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)
    clothes_tensor = transform(clothes).unsqueeze(0).to(device)
    edge_tensor = transform(edge).unsqueeze(0).to(device)

    # Run inference
    p_tryon, warped_cloth = pipeline(image_tensor, clothes_tensor, edge_tensor, phase="test")

    # Save the output image (adjust path as needed)
    output_path = "output.png"
    tv.utils.save_image(p_tryon, output_path, normalize=True, value_range=(-1, 1))

    return FileResponse(output_path)
