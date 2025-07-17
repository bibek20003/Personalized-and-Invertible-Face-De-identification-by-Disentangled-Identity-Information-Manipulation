from PIL import Image
import os

def make_comparison_image(original_path='/data/Bibek/Zace_deid_project/Input_Img/fake_class/000058.jpg', protected_path='/data/Bibek/protected.jpg', recovered_path='/data/Bibek/recovered.jpg', output_path='comparison.jpg'):
    images = []

    # Load images (resized to same height)
    def load_image(path):
        img = Image.open(path).convert("RGB")
        return img.resize((256, 256))

    if original_path and os.path.exists(original_path):
        images.append(load_image(original_path))
    else:
        print("⚠️ Skipping original image (not provided or not found)")

    if os.path.exists(protected_path):
        images.append(load_image(protected_path))
    else:
        print("❌ protected.jpg not found!")
        return

    if os.path.exists(recovered_path):
        images.append(load_image(recovered_path))
    else:
        print("❌ recovered.jpg not found!")
        return

    # Combine side-by-side
    widths, heights = zip(*(img.size for img in images))
    total_width = sum(widths)
    max_height = max(heights)

    comparison = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for img in images:
        comparison.paste(img, (x_offset, 0))
        x_offset += img.width

    # Save output
    comparison.save(output_path)
    print(f"✅ Saved comparison image to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--original', help='Path to original image (optional)', default=None)
    parser.add_argument('--protected', help='Path to protected image', default='protected.jpg')
    parser.add_argument('--recovered', help='Path to recovered image', default='recovered.jpg')
    parser.add_argument('--output', help='Output file name', default='comparison.jpg')
    args = parser.parse_args()

    make_comparison_image(args.original, args.protected, args.recovered, args.output)
