export type TourPlacement = "top" | "bottom" | "left" | "right" | "center";

export interface TourStep {
  id: string;
  title: string;
  content: string;
  target: string | null;
  placement: TourPlacement;
  icon: string;
  tip?: string;
  spotlightPadding?: number;
  spotlightRadius?: number;
}

export const tourSteps: TourStep[] = [
  {
    id: "welcome",
    title: "Welcome to Manuscript Layout Analysis",
    content:
      "This tool helps you detect and annotate layout elements in medieval manuscripts using YOLO models. Let me show you around the main features.",
    target: null,
    placement: "center",
    icon: "book-open",
  },
  {
    id: "file-upload",
    title: "Upload Your Manuscript",
    content:
      "Drag and drop images or ZIP archives here, or click to browse. Supported formats include PNG, JPG, TIFF, and WEBP.",
    target: '[data-tour="file-upload"]',
    placement: "right",
    icon: "upload",
    tip: "You can upload multiple images at once, or a ZIP archive containing nested folders.",
  },
  {
    id: "left-panel",
    title: "File Management",
    content:
      "Your uploaded files appear here. Each file shows its status, thumbnail, and per-file settings. Click a file to select it for viewing.",
    target: '[data-tour="left-panel"]',
    placement: "right",
    icon: "panel-left",
    spotlightPadding: 0,
  },
  {
    id: "class-filter",
    title: "Class Filter",
    content:
      "Select which manuscript element classes to detect. The model recognizes 25 classes including TextBlock, Decoration, Illustration, and more.",
    target: '[data-tour="class-filter"]',
    placement: "left",
    icon: "filter",
    tip: "Use Select All / Deselect All for quick toggling.",
  },
  {
    id: "detection-settings",
    title: "Detection Settings",
    content:
      "Adjust the confidence threshold and IoU (Intersection over Union) for each file. Higher confidence means fewer but more accurate detections.",
    target: '[data-tour="detection-settings"]',
    placement: "left",
    icon: "sliders-horizontal",
  },
  {
    id: "run-analysis",
    title: "Run Analysis",
    content:
      "Once you've uploaded files and selected classes, click here to run the YOLO model analysis on all pending files.",
    target: '[data-tour="run-analysis"]',
    placement: "bottom",
    icon: "play",
    spotlightPadding: 4,
    spotlightRadius: 6,
  },
  {
    id: "interactive-canvas",
    title: "Interactive Canvas",
    content:
      "After analysis, results appear here as an interactive canvas. Pan and zoom to explore, and click on annotations to inspect them.",
    target: '[data-tour="canvas"]',
    placement: "left",
    icon: "scan-eye",
    tip: "Use the scroll wheel to zoom, and drag to pan around the image.",
    spotlightPadding: 0,
  },
  {
    id: "results-tab",
    title: "Results & Statistics",
    content:
      "View detection statistics, class distribution charts, and counts for each detected element type.",
    target: '[data-tour="results-tab"]',
    placement: "left",
    icon: "bar-chart-3",
  },
  {
    id: "annotation-inspector",
    title: "Annotation Inspector",
    content:
      "Click any annotation on the canvas to see its details here: class name, bounding box coordinates, area, and segmentation mask.",
    target: '[data-tour="inspector-tab"]',
    placement: "left",
    icon: "info",
  },
  {
    id: "export",
    title: "Export Results",
    content:
      "Download your results in multiple formats: COCO JSON for further processing, annotated images, or a complete ZIP archive.",
    target: '[data-tour="export"]',
    placement: "bottom",
    icon: "download",
    spotlightPadding: 4,
    spotlightRadius: 6,
  },
  {
    id: "theme-toggle",
    title: "Theme Options",
    content:
      "Switch between light, dark, and high-contrast themes to suit your preference and working environment.",
    target: '[data-tour="theme-toggle"]',
    placement: "bottom",
    icon: "sun-moon",
  },
  {
    id: "complete",
    title: "You're All Set!",
    content:
      "That covers the basics! Start by uploading a manuscript image and running an analysis. You can restart this tour anytime from the help button in the top bar.",
    target: null,
    placement: "center",
    icon: "check-circle",
  },
];

export function getStepById(id: string): TourStep | undefined {
  return tourSteps.find((step) => step.id === id);
}

export function getStepIndexById(id: string): number {
  return tourSteps.findIndex((step) => step.id === id);
}
