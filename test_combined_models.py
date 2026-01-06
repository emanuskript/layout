#!/usr/bin/env python3
"""
Script to combine predictions from three YOLO models:
- best_emanuskript_segmentation.pt (segmentation model for manuscript elements)
- best_catmus.pt (segmentation model for lines and zones)
- best_zone_detection.pt (detection model for zones)

WORKFLOW SUMMARY:
================

1. MODEL PREDICTIONS (run_model_predictions):
   - Runs each of the 3 models on the input image
   - Saves predictions to JSON files in separate folders
   - Emanuskript: detects manuscript elements (Main script, Plain initial, etc.)
   - Catmus: detects lines (DefaultLine, InterlinearLine)
   - Zone: detects zones (MainZone, DropCapitalZone, etc.)

2. COMBINING & FILTERING (combine_and_filter_predictions):
   - Uses ImageBatch class to:
     a) Load all predictions from the 3 JSON files
     b) Unify class names (maps catmus/zone names to coco_class_mapping)
     c) Filter overlapping/conflicting annotations using spatial indexing
     d) Convert to COCO format
   - Only keeps classes defined in coco_class_mapping (25 classes total)

3. OUTPUT:
   - COCO format JSON file with filtered annotations
   - Only contains classes from coco_class_mapping

KEY CLASSES IN coco_class_mapping:
- Main script black/coloured
- Variant script black/coloured  
- Plain initial (coloured/highlighted/black)
- Historiated, Inhabited, Embellished
- Page Number, Quire Mark, Running header
- Gloss, Illustrations, Column
- Music, MusicZone, MusicLine
- Border, Table, Diagram
- GraphicZone

The ImageBatch class handles:
- Spatial overlap detection (removes duplicates)
- Class name unification (catmus_zones_mapping)
- Annotation filtering based on overlap thresholds
"""

import os
import json
import tempfile
import shutil
from pathlib import Path
from ultralytics import YOLO
import sys

# Add current directory to path to import ImageBatch
current_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = current_dir  # This file is in the project root
sys.path.insert(0, current_dir)

try:
    from utils.image_batch_classes import ImageBatch, coco_class_mapping
except ImportError as e:
    print(f"Warning: Could not import ImageBatch: {e}")
    print("Make sure all dependencies are installed (rtree, shapely, etc.)")
    ImageBatch = None

def run_model_predictions(image_path, output_dir):
    """Run all three models on the image and save predictions."""
    
    # Create output directories
    catmus_dir = os.path.join(output_dir, 'catmus')
    emanuskript_dir = os.path.join(output_dir, 'emanuskript')
    zone_dir = os.path.join(output_dir, 'zone')
    
    for dir_path in [catmus_dir, emanuskript_dir, zone_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    image_id = Path(image_path).stem
    
    print("=" * 60)
    print("Running Model Predictions")
    print("=" * 60)
    
    # 1. Emanuskript model
    print("\n[1/3] Running emanuskript segmentation model...")
    emanuskript_model_path = os.path.join(PROJECT_ROOT, "best_emanuskript_segmentation.pt")
    emanuskript_model = YOLO(emanuskript_model_path)
    emanuskript_classes = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20]
    emanuskript_results = emanuskript_model.predict(
        image_path,
        classes=emanuskript_classes,
        iou=0.3,
        device='cpu',
        augment=False,
        stream=False
    )
    emanuskript_path = f'{emanuskript_dir}/{image_id}.json'
    with open(emanuskript_path, 'w') as f:
        f.write(emanuskript_results[0].to_json())
    print(f"  ✓ Saved to: {emanuskript_path}")
    del emanuskript_model
    del emanuskript_results
    
    # 2. Catmus model
    print("\n[2/3] Running catmus segmentation model...")
    catmus_model_path = os.path.join(PROJECT_ROOT, "best_catmus.pt")
    catmus_model = YOLO(catmus_model_path)
    catmus_classes = [1, 7]  # DefaultLine and InterlinearLine
    catmus_results = catmus_model.predict(
        image_path,
        classes=catmus_classes,
        iou=0.3,
        device='cpu',
        augment=False,
        stream=False
    )
    catmus_path = f'{catmus_dir}/{image_id}.json'
    with open(catmus_path, 'w') as f:
        f.write(catmus_results[0].to_json())
    print(f"  ✓ Saved to: {catmus_path}")
    del catmus_model
    del catmus_results
    
    # 3. Zone detection model
    print("\n[3/3] Running zone detection model...")
    zone_model_path = os.path.join(PROJECT_ROOT, "best_zone_detection.pt")
    zone_model = YOLO(zone_model_path)
    zone_results = zone_model.predict(
        image_path,
        device='cpu',
        iou=0.3,
        augment=False,
        stream=False
    )
    zone_path = f'{zone_dir}/{image_id}.json'
    with open(zone_path, 'w') as f:
        f.write(zone_results[0].to_json())
    print(f"  ✓ Saved to: {zone_path}")
    del zone_model
    del zone_results
    
    return {
        'catmus': catmus_dir,
        'emanuskript': emanuskript_dir,
        'zone': zone_dir
    }


def combine_and_filter_predictions(image_path, labels_folders, output_json_path=None):
    """Combine predictions from all models and filter to coco_class_mapping classes."""
    
    print("\n" + "=" * 60)
    print("Combining and Filtering Predictions")
    print("=" * 60)
    
    if ImageBatch is None:
        print("\nERROR: ImageBatch class not available.")
        print("Please install missing dependencies:")
        print("  pip install rtree shapely")
        return None
    
    # Create a temporary folder with just the image file
    # ImageBatch.load_images() loads all files in the folder, so we need only images
    temp_image_dir = tempfile.mkdtemp()
    image_filename = os.path.basename(image_path)
    temp_image_path = os.path.join(temp_image_dir, image_filename)
    shutil.copy2(image_path, temp_image_path)
    
    # Create ImageBatch instance
    image_folder = temp_image_dir
    
    image_batch = ImageBatch(
        image_folder=image_folder,
        catmus_labels_folder=labels_folders['catmus'],
        emanuskript_labels_folder=labels_folders['emanuskript'],
        zone_labels_folder=labels_folders['zone']
    )
    
    # Load images
    print("\n[Step 1] Loading images...")
    image_batch.load_images()
    print(f"  ✓ Loaded {len(image_batch.images)} image(s)")
    
    # Load annotations from all three models
    print("\n[Step 2] Loading annotations from all models...")
    image_batch.load_annotations()
    
    total_annotations = sum(len(img.annotations) for img in image_batch.images)
    print(f"  ✓ Loaded {total_annotations} total annotations")
    
    # Unify names (maps catmus/zone names to coco_class_mapping names)
    print("\n[Step 3] Unifying class names...")
    image_batch.unify_names()
    
    # Filter annotations (removes overlapping/conflicting annotations)
    print("\n[Step 4] Filtering annotations...")
    for img in image_batch.images:
        filtered = img.filter_annotations()
        print(f"  Image {img.filename}: {len(img.annotations)} -> {len(filtered)} annotations")
    
    # Get COCO format JSON
    print("\n[Step 5] Generating COCO format...")
    coco_json = image_batch.return_coco_file()
    
    # Filter to only classes in coco_class_mapping
    valid_category_ids = set(coco_class_mapping.values())
    
    filtered_annotations = [
        ann for ann in coco_json['annotations']
        if ann['category_id'] in valid_category_ids
    ]
    
    coco_json['annotations'] = filtered_annotations
    
    # Update categories to only include valid ones
    coco_json['categories'] = [
        cat for cat in coco_json['categories']
        if cat['id'] in valid_category_ids
    ]
    
    print(f"  ✓ Final annotations: {len(filtered_annotations)}")
    print(f"  ✓ Final categories: {len(coco_json['categories'])}")
    
    # Save to file if path provided
    if output_json_path:
        with open(output_json_path, 'w') as f:
            json.dump(coco_json, f, indent=2)
        print(f"\n  ✓ Saved COCO JSON to: {output_json_path}")
    
    # Cleanup temporary image directory
    shutil.rmtree(temp_image_dir, ignore_errors=True)
    
    return coco_json


def print_summary(coco_json):
    """Print summary of results."""
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    
    # Category counts
    category_counts = {}
    for ann in coco_json['annotations']:
        cat_id = ann['category_id']
        category_counts[cat_id] = category_counts.get(cat_id, 0) + 1
    
    # Map category IDs to names
    id_to_name = {cat['id']: cat['name'] for cat in coco_json['categories']}
    
    print(f"\nTotal Annotations: {len(coco_json['annotations'])}")
    print(f"Total Categories: {len(coco_json['categories'])}")
    print(f"\nAnnotations per Category:")
    for cat_id in sorted(category_counts.keys()):
        name = id_to_name.get(cat_id, f"Unknown({cat_id})")
        count = category_counts[cat_id]
        print(f"  {name:30s}: {count:4d}")


def visualize_results(image_path, coco_json):
    """Visualize the combined results on the image."""
    print("\n" + "=" * 60)
    print("Visualizing Results")
    print("=" * 60)
    
    try:
        from utils.image_batch_classes import ImageBatch
        import tempfile
        
        # Create temporary labels folders for visualization
        with tempfile.TemporaryDirectory() as temp_dir:
            # We need to recreate the ImageBatch with the combined results
            # For now, just show the COCO JSON structure
            print("\nTo visualize, you can:")
            print("1. Use the COCO JSON file with any COCO visualization tool")
            print("2. Load the JSON in your annotation tool")
            print("3. Use the ImageBatch.plot_annotations() method")
            
    except Exception as e:
        print(f"Visualization not available: {e}")


def main():
    """Main function to run the complete pipeline."""
    
    # Configuration
    image_path = "bnf-naf-10039__page-001-of-004.jpg"
    output_json = "combined_predictions.json"
    
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return
    
    # Create temporary directory for predictions
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")
        
        # Step 1: Run all three models
        labels_folders = run_model_predictions(image_path, temp_dir)
        
        # Step 2: Combine and filter predictions
        coco_json = combine_and_filter_predictions(
            image_path,
            labels_folders,
            output_json_path=output_json
        )
        
        # Step 3: Print summary
        print_summary(coco_json)
        
        # Step 4: Visualize (optional)
        visualize_results(image_path, coco_json)
        
        print("\n" + "=" * 60)
        print("Pipeline Complete!")
        print("=" * 60)
        print(f"\nOutput saved to: {output_json}")


if __name__ == "__main__":
    main()

