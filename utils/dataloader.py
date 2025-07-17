from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def get_dataloader(data_root, image_size=256, batch_size=16):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])
    
    dataset = ImageFolder(root=data_root, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
