from typing import Tuple, Dict, List, Union
import gradio as gr
import supervision as sv
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import zipfile
import os
import tempfile
import cv2
import json
from datetime import datetime
import io

# Define custom models
MODEL_FILES = {
    "Line Detection": "best_line_detection_yoloe (1).pt",
    "Border Detection": "border_model_weights.pt", 
    "Zones Detection": "zones_model_weights.pt"
}

# Dictionary to store loaded models
models: Dict[str, YOLO] = {}

# Global variables to store results for download
current_results = []
current_images = []

# Load all custom models
for name, model_file in MODEL_FILES.items():
    model_path = os.path.join(os.getcwd(), model_file)
    if os.path.exists(model_path):
        try:
            models[name] = YOLO(model_path)
            print(f"✓ Loaded {name} model from {model_path}")
        except Exception as e:
            print(f"✗ Error loading {name} model: {e}")
            models[name] = None
    else:
        print(f"✗ Warning: Model file {model_path} not found")
        models[name] = None

# Create annotators
LABEL_ANNOTATOR = sv.LabelAnnotator(text_color=sv.Color.BLACK)
BOX_ANNOTATOR = sv.BoxAnnotator()

def detect_and_annotate_combined(
    image: np.ndarray,
    conf_threshold: float,
    iou_threshold: float,
    return_annotations: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
    """Run all three models and combine their outputs in a single annotated image"""
    print(f"🔍 Starting detection on image shape: {image.shape}")
    
    # Colors for different models - more distinct colors
    colors = {
        "Line Detection": sv.Color.from_hex("#FF0000"),      # Bright Red
        "Border Detection": sv.Color.from_hex("#00FF00"),   # Bright Green  
        "Zones Detection": sv.Color.from_hex("#0080FF")     # Bright Blue
    }
    
    # Model prefixes for clear labeling
    model_prefixes = {
        "Line Detection": "[LINE]",
        "Border Detection": "[BORDER]", 
        "Zones Detection": "[ZONE]"
    }
    
    annotated_image = image.copy()
    total_detections = 0
    detections_data = {}
    
    # Run each model and annotate with different colors
    for model_name, model in models.items():
        if model is None:
            print(f"⏭️  Skipping {model_name} (model not loaded)")
            detections_data[model_name] = []
            continue
            
        print(f"🤖 Running {model_name} model...")
        
        # Perform inference
        results = model.predict(
            image,
            conf=conf_threshold,
            iou=iou_threshold
        )[0]
        
        model_detections = []
        
        if len(results.boxes) > 0:
            print(f"   Found {len(results.boxes)} detections")
            total_detections += len(results.boxes)
            
            # Convert results to supervision Detections
            boxes = results.boxes.xyxy.cpu().numpy()
            confidence = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy().astype(int)
            
            # Store detection data for COCO format
            for i, (box, conf, class_id) in enumerate(zip(boxes, confidence, class_ids)):
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1
                
                model_detections.append({
                    "bbox": [float(x1), float(y1), float(width), float(height)],  # COCO format: [x, y, width, height]
                    "class_name": results.names[class_id],
                    "confidence": float(conf)
                })
            
            # Create Detections object for visualization
            detections = sv.Detections(
                xyxy=boxes,
                confidence=confidence,
                class_id=class_ids
            )
            
            # Create labels with clear model prefixes and confidence scores
            model_prefix = model_prefixes[model_name]
            labels = [
                f"{model_prefix} {results.names[class_id]} ({conf:.2f})"
                for class_id, conf
                in zip(class_ids, confidence)
            ]

            # Create annotators with specific colors and improved styling
            box_annotator = sv.BoxAnnotator(
                color=colors[model_name],
                thickness=3  # Thicker boxes for better visibility
            )
            label_annotator = sv.LabelAnnotator(
                text_color=sv.Color.WHITE,
                color=colors[model_name],
                text_thickness=2,
                text_scale=0.6,
                text_padding=8
            )
            
            # Annotate image
            annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)
            annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
            print(f"   ✅ Annotated with {len(boxes)} {model_name} detections")
        else:
            print(f"   No detections found for {model_name}")
        
        detections_data[model_name] = model_detections
    
    print(f"🎯 Detection completed. Total detections: {total_detections}")
    
    if return_annotations:
        return annotated_image, detections_data
    else:
        return annotated_image

def process_zip_file(zip_file_path: str, conf_threshold: float, iou_threshold: float) -> Tuple[List[Tuple[str, np.ndarray]], List[Tuple[str, Dict]], Dict]:
    """Process all images in a zip file and return annotated images, detection data, and image info"""
    print(f"📁 Opening ZIP file: {zip_file_path}")
    results = []
    annotations_data = []
    image_info = {}
    
    try:
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            print(f"📋 ZIP file contents: {zip_ref.namelist()}")
            
            # Create temporary directory to extract files
            with tempfile.TemporaryDirectory() as temp_dir:
                print(f"📂 Extracting to temporary directory: {temp_dir}")
                zip_ref.extractall(temp_dir)
                
                # List all files in temp directory
                all_files = os.listdir(temp_dir)
                print(f"📄 Files extracted: {all_files}")
                
                # Process each image file (recursively search through folders)
                image_count = 0
                
                # Walk through all directories and subdirectories
                for root, dirs, files in os.walk(temp_dir):
                    print(f"📂 Searching in directory: {root}")
                    
                    for filename in files:
                        # Skip macOS hidden files
                        if filename.startswith('._') or filename.startswith('.DS_Store'):
                            print(f"⏭️  Skipping system file: {filename}")
                            continue
                            
                        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                            image_count += 1
                            image_path = os.path.join(root, filename)
                            print(f"🖼️  Processing image {image_count}: {filename} (from {os.path.relpath(root, temp_dir)})")
                            
                            # Load image
                            image = cv2.imread(image_path)
                            if image is not None:
                                print(f"✅ Image loaded successfully: {image.shape}")
                                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                                
                                # Store image info
                                height, width = image.shape[:2]
                                image_info[filename] = (height, width)
                                
                                # Process with all models and get annotation data
                                print(f"🔍 Running detection models on {filename}...")
                                annotated_image, detections_data = detect_and_annotate_combined(
                                    image, conf_threshold, iou_threshold, return_annotations=True
                                )
                                print(f"✅ Detection completed for {filename}")
                                
                                results.append((filename, annotated_image))
                                annotations_data.append((filename, detections_data))
                            else:
                                print(f"❌ Failed to load image: {filename}")
                        else:
                            print(f"⏭️  Skipping non-image file: {filename}")
                
                print(f"📊 Total images processed: {len(results)} out of {image_count} image files found")
                print(f"📁 Searched through all subdirectories recursively")
        
        print(f"🎉 ZIP processing completed successfully! Processed {len(results)} images")
        return results, annotations_data, image_info
        
    except Exception as e:
        print(f"💥 ERROR in process_zip_file: {str(e)}")
        import traceback
        traceback.print_exc()
        return [], [], {}

def create_coco_annotations(results_data: List, image_info: Dict) -> Dict:
    """Convert detection results to COCO JSON format"""
    coco_data = {
        "info": {
            "description": "Medieval Manuscript Detection Results",
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "Medieval YOLO Models",
            "date_created": datetime.now().isoformat()
        },
        "licenses": [
            {
                "id": 1,
                "name": "Custom License",
                "url": ""
            }
        ],
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Create categories from all models
    category_id = 1
    category_map = {}
    
    # Add categories for each model type
    for model_name in ["Line Detection", "Border Detection", "Zones Detection"]:
        if model_name in models and models[model_name] is not None:
            model = models[model_name]
            for class_id, class_name in model.names.items():
                full_name = f"{model_name}_{class_name}"
                if full_name not in category_map:
                    category_map[full_name] = category_id
                    coco_data["categories"].append({
                        "id": category_id,
                        "name": full_name,
                        "supercategory": model_name
                    })
                    category_id += 1
    
    annotation_id = 1
    
    for image_idx, (filename, detections_by_model) in enumerate(results_data):
        # Add image info
        image_id = image_idx + 1
        img_height, img_width = image_info.get(filename, (0, 0))
        
        coco_data["images"].append({
            "id": image_id,
            "file_name": filename,
            "width": img_width,
            "height": img_height,
            "license": 1
        })
        
        # Add annotations for each model
        for model_name, detections in detections_by_model.items():
            if detections:
                for detection in detections:
                    bbox = detection["bbox"]  # [x, y, width, height]
                    class_name = detection["class_name"]
                    confidence = detection["confidence"]
                    
                    full_category_name = f"{model_name}_{class_name}"
                    category_id = category_map.get(full_category_name, 1)
                    
                    coco_data["annotations"].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": bbox,
                        "area": bbox[2] * bbox[3],
                        "iscrowd": 0,
                        "score": confidence
                    })
                    annotation_id += 1
    
    return coco_data

def create_download_zip(images: List[Tuple[str, np.ndarray]], annotations: Dict) -> str:
    """Create a ZIP file with images and annotations"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = f"medieval_detection_results_{timestamp}.zip"
    zip_path = os.path.join(tempfile.gettempdir(), zip_filename)
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add images
        for filename, image_array in images:
            # Convert numpy array to PIL Image and save as bytes
            pil_image = Image.fromarray(image_array.astype('uint8'))
            img_bytes = io.BytesIO()
            
            # Determine format from filename
            if filename.lower().endswith('.png'):
                pil_image.save(img_bytes, format='PNG')
            else:
                pil_image.save(img_bytes, format='JPEG')
            
            # Add to ZIP
            zipf.writestr(f"images/{filename}", img_bytes.getvalue())
        
        # Add annotations
        annotations_json = json.dumps(annotations, indent=2)
        zipf.writestr("annotations.json", annotations_json)
        
        # Add README
        readme_content = f"""Medieval Manuscript Detection Results
=============================================

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Contents:
- images/: Annotated images with detection results
- annotations.json: COCO format annotations

Models and Color Coding:
- Line Detection (Red boxes with [LINE] prefix)
- Border Detection (Green boxes with [BORDER] prefix) 
- Zones Detection (Blue boxes with [ZONE] prefix)

Label format: [MODEL] class_name (confidence_score)
Annotation format: COCO JSON
For more info: https://cocodataset.org/#format-data
"""
        zipf.writestr("README.txt", readme_content)
    
    return zip_path

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Medieval Manuscript Detection with Custom YOLO Models")
    gr.Markdown("""
    **Models and Color Coding:**
    - 🔴 **Line Detection** - Red boxes with [LINE] prefix
    - 🟢 **Border Detection** - Green boxes with [BORDER] prefix  
    - 🔵 **Zones Detection** - Blue boxes with [ZONE] prefix
    
    Each detection shows: **[MODEL] class_name (confidence_score)**
    """)
    
    with gr.Tabs():
        # Single Image Tab
        with gr.TabItem("Single Image"):
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(
                        label="Input Image",
                        type='numpy'
                    )
                    with gr.Accordion("Detection Settings", open=True):
                        with gr.Row():
                            conf_threshold = gr.Slider(
                                label="Confidence Threshold",
                                minimum=0.0,
                                maximum=1.0,
                                step=0.05,
                                value=0.25,
                            )
                            iou_threshold = gr.Slider(
                                label="IoU Threshold",
                                minimum=0.0,
                                maximum=1.0,
                                step=0.05,
                                value=0.45,
                                info="Decrease for stricter detection, increase for more overlapping boxes"
                            )
                    with gr.Row():
                        clear_btn = gr.Button("Clear")
                        detect_btn = gr.Button("Detect with All Models", variant="primary")
                        
                with gr.Column():
                    output_image = gr.Image(
                        label="Combined Detection Result",
                        type='numpy'
                    )
                    
                    # Single image download buttons
                    with gr.Row():
                        single_download_json_btn = gr.Button(
                            "📄 Download Annotations (JSON)",
                            variant="secondary",
                            size="sm"
                        )
                        single_download_image_btn = gr.Button(
                            "🖼️ Download Image",
                            variant="secondary",
                            size="sm"
                        )
                    
                    # Single image file outputs
                    single_json_output = gr.File(
                        label="📄 JSON Download",
                        visible=True,
                        height=50
                    )
                    single_image_output = gr.File(
                        label="🖼️ Image Download",
                        visible=True,
                        height=50
                    )
        
        # Batch Processing Tab
        with gr.TabItem("Batch Processing (ZIP)"):
            with gr.Row():
                with gr.Column():
                    zip_file = gr.File(
                        label="Upload ZIP file with images",
                        file_types=[".zip"]
                    )
                    with gr.Accordion("Detection Settings", open=True):
                        with gr.Row():
                            batch_conf_threshold = gr.Slider(
                                label="Confidence Threshold",
                                minimum=0.0,
                                maximum=1.0,
                                step=0.05,
                                value=0.25,
                            )
                            batch_iou_threshold = gr.Slider(
                                label="IoU Threshold",
                                minimum=0.0,
                                maximum=1.0,
                                step=0.05,
                                value=0.45,
                            )
                    
                    # Add status message box
                    batch_status = gr.Textbox(
                        label="Processing Status",
                        value="Ready to process ZIP file...",
                        interactive=False,
                        max_lines=3
                    )
                    
                    with gr.Row():
                        clear_batch_btn = gr.Button("Clear")
                        process_batch_btn = gr.Button("Process ZIP", variant="primary")
                        
                with gr.Column():
                    batch_gallery = gr.Gallery(
                        label="Batch Processing Results",
                        show_label=True,
                        elem_id="gallery",
                        columns=2,
                        rows=2,
                        height="auto",
                        type="numpy"  # Explicitly handle numpy arrays
                    )
                    
                    # Download buttons
                    with gr.Row():
                        download_json_btn = gr.Button(
                            "📄 Download COCO Annotations (JSON)",
                            variant="secondary"
                        )
                        download_zip_btn = gr.Button(
                            "📦 Download Results (ZIP)",
                            variant="secondary"
                        )
                    
                    # File outputs for downloads
                    json_file_output = gr.File(
                        label="📄 JSON Download",
                        visible=True,
                        height=50
                    )
                    zip_file_output = gr.File(
                        label="📦 ZIP Download",
                        visible=True,
                        height=50
                    )

    # Global variables for single image results
    single_image_result = None
    single_image_annotations = None
    single_image_filename = None
    
    def process_single_image(
        image: np.ndarray,
        conf_threshold: float,
        iou_threshold: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        global single_image_result, single_image_annotations, single_image_filename
        
        if image is None:
            single_image_result = None
            single_image_annotations = None
            single_image_filename = None
            return None, None
            
        # Process with annotations
        annotated_image, detections_data = detect_and_annotate_combined(
            image, conf_threshold, iou_threshold, return_annotations=True
        )
        
        # Store results globally for download
        single_image_result = annotated_image
        single_image_annotations = detections_data
        single_image_filename = f"detection_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        
        return image, annotated_image

    def process_batch_images_with_status(
        zip_file,
        conf_threshold: float,
        iou_threshold: float
    ):
        print("🚀 ========== BATCH PROCESSING STARTED ==========")
        
        if zip_file is None:
            print("❌ No ZIP file provided")
            return [], "Please upload a ZIP file first."
        
        print(f"📁 ZIP file received: {zip_file.name}")
        print(f"⚙️  Settings: conf_threshold={conf_threshold}, iou_threshold={iou_threshold}")
        
        try:
            # Process zip file
            print("🔄 Starting ZIP file processing...")
            results, annotations_data, image_info = process_zip_file(zip_file.name, conf_threshold, iou_threshold)
            
            if not results:
                error_msg = "No valid images found in ZIP file."
                print(f"❌ {error_msg}")
                return [], error_msg
            
            # Store data globally for download
            global current_results, current_images
            current_images = results
            current_results = annotations_data
            
            print(f"📊 ZIP processing returned {len(results)} results")
            
            # Convert results to format expected by Gallery
            print("🔄 Converting results for Gradio Gallery...")
            gallery_images = []
            
            for i, (filename, annotated_image) in enumerate(results):
                print(f"🖼️  Converting image {i+1}/{len(results)}: {filename}")
                print(f"   Image shape: {annotated_image.shape}, dtype: {annotated_image.dtype}")
                
                # Ensure the image is in the right format and range
                if annotated_image.dtype != 'uint8':
                    print(f"   Converting dtype from {annotated_image.dtype} to uint8")
                    # Normalize if needed
                    if annotated_image.max() <= 1.0:
                        annotated_image = (annotated_image * 255).astype('uint8')
                        print(f"   Normalized from [0,1] to [0,255]")
                    else:
                        annotated_image = annotated_image.astype('uint8')
                        print(f"   Cast to uint8")
                
                print(f"   Final image shape: {annotated_image.shape}, dtype: {annotated_image.dtype}")
                
                # For Gradio gallery, we can pass numpy arrays directly
                # Format: (image_data, caption)
                gallery_images.append((annotated_image, filename))
                print(f"   ✅ Added {filename} to gallery")
            
            success_msg = f"✅ Successfully processed {len(gallery_images)} images!"
            print(f"🎉 {success_msg}")
            print(f"📋 Gallery contains {len(gallery_images)} items")
            print("🏁 ========== BATCH PROCESSING COMPLETED ==========\n")
            
            return gallery_images, success_msg
            
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            print(f"💥 EXCEPTION in process_batch_images_with_status: {error_msg}")
            import traceback
            traceback.print_exc()
            print("💀 ========== BATCH PROCESSING FAILED ==========\n")
            return [], error_msg

    def clear_single():
        return None, None
    
    def clear_batch():
        global current_results, current_images
        current_results = []
        current_images = []
        return None, [], "Ready to process ZIP file..."
    
    def download_annotations():
        """Create and return COCO JSON annotations file"""
        global current_results, current_images
        
        if not current_results:
            print("❌ No annotation data available for download")
            return None
        
        try:
            # Create image info dictionary
            image_info = {}
            for filename, image_array in current_images:
                height, width = image_array.shape[:2]
                image_info[filename] = (height, width)
            
            # Create COCO annotations
            coco_data = create_coco_annotations(current_results, image_info)
            
            # Save to temporary file with proper name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            json_filename = f"medieval_annotations_{timestamp}.json"
            json_path = os.path.join(tempfile.gettempdir(), json_filename)
            
            with open(json_path, 'w') as f:
                json.dump(coco_data, f, indent=2)
            
            print(f"💾 Created annotations file: {json_path}")
            print(f"📁 File size: {os.path.getsize(json_path)} bytes")
            
            # Verify file exists and is readable
            if os.path.exists(json_path) and os.path.getsize(json_path) > 0:
                return json_path
            else:
                print(f"❌ File verification failed: {json_path}")
                return None
            
        except Exception as e:
            print(f"❌ Error creating annotations: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def download_results_zip():
        """Create and return ZIP file with images and annotations"""
        global current_results, current_images
        
        if not current_results or not current_images:
            print("❌ No results data available for ZIP download")
            return None
        
        try:
            # Create image info dictionary
            image_info = {}
            for filename, image_array in current_images:
                height, width = image_array.shape[:2]
                image_info[filename] = (height, width)
            
            # Create COCO annotations
            coco_data = create_coco_annotations(current_results, image_info)
            
            # Create ZIP file
            zip_path = create_download_zip(current_images, coco_data)
            
            print(f"💾 Created results ZIP: {zip_path}")
            print(f"📁 ZIP file size: {os.path.getsize(zip_path)} bytes")
            
            # Verify file exists and is readable
            if os.path.exists(zip_path) and os.path.getsize(zip_path) > 0:
                return zip_path
            else:
                print(f"❌ ZIP file verification failed: {zip_path}")
                return None
            
        except Exception as e:
            print(f"❌ Error creating ZIP file: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def download_single_annotations():
        """Download COCO annotations for single image"""
        global single_image_annotations, single_image_result, single_image_filename
        
        if single_image_annotations is None or single_image_result is None:
            print("❌ No single image annotation data available")
            return None
        
        try:
            # Create image info
            height, width = single_image_result.shape[:2]
            image_info = {single_image_filename: (height, width)}
            
            # Create annotations data in the expected format
            annotations_data = [(single_image_filename, single_image_annotations)]
            
            # Create COCO annotations
            coco_data = create_coco_annotations(annotations_data, image_info)
            
            # Save to temporary file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            json_filename = f"single_image_annotations_{timestamp}.json"
            json_path = os.path.join(tempfile.gettempdir(), json_filename)
            
            with open(json_path, 'w') as f:
                json.dump(coco_data, f, indent=2)
            
            print(f"💾 Created single image annotations: {json_path}")
            print(f"📁 File size: {os.path.getsize(json_path)} bytes")
            
            # Verify file exists
            if os.path.exists(json_path) and os.path.getsize(json_path) > 0:
                return json_path
            else:
                print(f"❌ Single image file verification failed: {json_path}")
                return None
            
        except Exception as e:
            print(f"❌ Error creating single image annotations: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def download_single_image():
        """Download processed single image"""
        global single_image_result, single_image_filename
        
        if single_image_result is None:
            print("❌ No single image result available")
            return None
        
        try:
            # Convert to PIL and save
            pil_image = Image.fromarray(single_image_result.astype('uint8'))
            
            # Save to temporary file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            img_filename = f"processed_image_{timestamp}.jpg"
            img_path = os.path.join(tempfile.gettempdir(), img_filename)
            
            pil_image.save(img_path, 'JPEG', quality=95)
            
            print(f"💾 Created single image file: {img_path}")
            print(f"📁 Image file size: {os.path.getsize(img_path)} bytes")
            
            # Verify file exists
            if os.path.exists(img_path) and os.path.getsize(img_path) > 0:
                return img_path
            else:
                print(f"❌ Single image file verification failed: {img_path}")
                return None
            
        except Exception as e:
            print(f"❌ Error creating single image file: {e}")
            import traceback
            traceback.print_exc()
            return None

    # Connect buttons to functions for single image
    detect_btn.click(
        process_single_image,
        inputs=[input_image, conf_threshold, iou_threshold],
        outputs=[input_image, output_image]
    )
    clear_btn.click(
        clear_single,
        inputs=None,
        outputs=[input_image, output_image]
    )
    
    # Connect buttons to functions for batch processing
    process_batch_btn.click(
        process_batch_images_with_status,
        inputs=[zip_file, batch_conf_threshold, batch_iou_threshold],
        outputs=[batch_gallery, batch_status]
    )
    clear_batch_btn.click(
        clear_batch,
        inputs=None,
        outputs=[zip_file, batch_gallery, batch_status]
    )
    
    # Connect download buttons
    download_json_btn.click(
        fn=download_annotations,
        inputs=[],
        outputs=[json_file_output]
    )
    download_zip_btn.click(
        fn=download_results_zip,
        inputs=[],
        outputs=[zip_file_output]
    )
    
    # Connect single image download buttons
    single_download_json_btn.click(
        fn=download_single_annotations,
        inputs=[],
        outputs=[single_json_output]
    )
    single_download_image_btn.click(
        fn=download_single_image,
        inputs=[],
        outputs=[single_image_output]
    )

if __name__ == "__main__":
    demo.launch(debug=True, show_error=True)
