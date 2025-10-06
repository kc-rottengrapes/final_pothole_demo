# Enhanced Pothole Detection with Counting and Renaming
from ultralytics import YOLO
import glob
import os
import cv2
from datetime import datetime

# Load your model
model = YOLO('best.pt')

# Ask user directly for folder path
choice = input("Enter folder path containing images: ").strip()

# Validate path
if not os.path.exists(choice):
    print(f"❌ Error: Folder '{choice}' does not exist.")
    exit(1)

# Get all images
image_files = glob.glob(f"{choice}/*.jpg") + glob.glob(f"{choice}/*.png") + glob.glob(f"{choice}/*.jpeg")
print(f"Found {len(image_files)} images in {choice}")

# Ask for confidence
conf = float(input("Enter confidence threshold (0.1-0.9, default 0.3): ") or 0.3)

# Create single output folder
output_dir = "pothole_results"
os.makedirs(output_dir, exist_ok=True)

print(f"\nProcessing {len(image_files)} images with confidence {conf}...")

# Initialize tracking variables
detection_results = []
total_potholes = 0
images_with_potholes = 0

for i, img_path in enumerate(image_files, 1):
    # Create standardized name
    original_name = os.path.basename(img_path)
    new_name = f"image_{i:03d}.jpg"
    
    # Run detection and save annotated image directly to output folder
    results = model(img_path, conf=conf, save=False, verbose=False)
    
    # Count potholes
    pothole_count = len(results[0].boxes)
    total_potholes += pothole_count
    
    if pothole_count > 0:
        images_with_potholes += 1
        print(f"✓ {new_name}: {pothole_count} potholes detected")
    else:
        print(f"✗ {new_name}: No potholes detected")
    
    # Save annotated image with new name
    annotated_img = results[0].plot()
    import cv2
    cv2.imwrite(os.path.join(output_dir, new_name), annotated_img)
    
    # Store results
    detection_results.append({
        'original_name': original_name,
        'renamed_name': new_name,
        'pothole_count': pothole_count
    })

# Generate summary report in same folder
summary_file = os.path.join(output_dir, "detection_summary.txt")
with open(summary_file, 'w') as f:
    f.write("POTHOLE DETECTION SUMMARY REPORT\n")
    f.write("=" * 40 + "\n\n")
    f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Confidence Threshold: {conf}\n\n")
    
    f.write("OVERALL STATISTICS:\n")
    f.write("-" * 20 + "\n")
    f.write(f"Total Images Processed: {len(image_files)}\n")
    f.write(f"Images with Potholes: {images_with_potholes}\n")
    f.write(f"Images without Potholes: {len(image_files) - images_with_potholes}\n")
    f.write(f"Total Potholes Detected: {total_potholes}\n")
    f.write(f"Average Potholes per Image: {total_potholes/len(image_files):.2f}\n\n")
    
    f.write("DETAILED RESULTS:\n")
    f.write("-" * 20 + "\n")
    for result in detection_results:
        f.write(f"{result['renamed_name']} (original: {result['original_name']}): {result['pothole_count']} potholes\n")

print(f"\n📊 FINAL SUMMARY:")
print(f"Total Images: {len(image_files)}")
print(f"Images with Potholes: {images_with_potholes}")
print(f"Total Potholes Found: {total_potholes}")
print(f"Average per Image: {total_potholes/len(image_files):.2f}")
print(f"\n📁 All results in single folder: {output_dir}")
print(f"📄 Summary report: {summary_file}")
print(f"🖼️ Annotated images: image_001.jpg, image_002.jpg, etc.")
