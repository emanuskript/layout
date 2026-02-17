/**
 * Subcategory groupings and display name mapping for the 22 manuscript
 * element classes, matching the Mergen annotation specification.
 *
 * Backend class names are kept unchanged for model/COCO compatibility.
 * The `displayNames` map converts them to the canonical PDF labels.
 */

export interface ClassCategory {
  name: string;
  /** Backend class names belonging to this subcategory */
  classes: string[];
}

export const classCategories: ClassCategory[] = [
  {
    name: "Layout",
    classes: ["Column"],
  },
  {
    name: "Main Text",
    classes: [
      "Main script black",
      "Main script coloured",
      "Variant script black",
      "Variant script coloured",
    ],
  },
  {
    name: "Initials",
    classes: [
      "Plain initial - Black",
      "Plain initial - Highlighted",
      "Plain initial- coloured",
      "Embellished",
      "Inhabited",
      "Historiated",
      "Zoo - Anthropomorphic",
    ],
  },
  {
    name: "Non-textual Elements",
    classes: [
      "Border",
      "Illustrations",
      "Table",
      "Diagram",
      "Music",
    ],
  },
  {
    name: "Paratextual Elements",
    classes: [
      "Running header",
      "Gloss",
      "Page Number",
      "Quire Mark",
      "Catchword",
    ],
  },
];

/** Backend class name → PDF display label (only entries that differ) */
export const displayNames: Record<string, string> = {
  "Zoo - Anthropomorphic": "ZooAnthromorphic",
  "Plain initial- coloured": "Plain initial \u2013 coloured",
  "Plain initial - Highlighted": "Plain initial \u2013 highlighted",
  "Plain initial - Black": "Plain initial \u2013 black",
  "Illustrations": "Illustration",
  "Page Number": "Page number",
  "Quire Mark": "Quire mark",
};

/** Get the display label for a backend class name. */
export function getDisplayName(backendName: string): string {
  return displayNames[backendName] ?? backendName;
}
