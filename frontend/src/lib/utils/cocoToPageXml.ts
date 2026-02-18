import { COCOJson } from "../types/coco";

const PAGE_NS =
  "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15";
const SCHEMA_LOC = `${PAGE_NS} http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15/pagecontent.xsd`;

/** Map 22 manuscript classes → [PAGE region element, optional @type] */
const REGION_MAP: Record<string, [string, string | null]> = {
  Border: ["SeparatorRegion", null],
  Table: ["TableRegion", null],
  Diagram: ["GraphicRegion", null],
  "Main script black": ["TextRegion", "paragraph"],
  "Main script coloured": ["TextRegion", "paragraph"],
  "Variant script black": ["TextRegion", "other"],
  "Variant script coloured": ["TextRegion", "other"],
  Historiated: ["ImageRegion", null],
  Inhabited: ["ImageRegion", null],
  "Zoo - Anthropomorphic": ["ImageRegion", null],
  Embellished: ["ImageRegion", null],
  "Plain initial- coloured": ["TextRegion", "drop-capital"],
  "Plain initial - Highlighted": ["TextRegion", "drop-capital"],
  "Plain initial - Black": ["TextRegion", "drop-capital"],
  "Page Number": ["TextRegion", "page-number"],
  "Quire Mark": ["TextRegion", "signature-mark"],
  "Running header": ["TextRegion", "header"],
  Catchword: ["TextRegion", "catch-word"],
  Gloss: ["TextRegion", "marginalia"],
  Illustrations: ["ImageRegion", null],
  Column: ["TextRegion", "paragraph"],
  Music: ["MusicRegion", null],
};

function segToPoints(seg: number[]): string {
  const parts: string[] = [];
  for (let i = 0; i < seg.length - 1; i += 2) {
    parts.push(`${Math.round(seg[i])},${Math.round(seg[i + 1])}`);
  }
  return parts.join(" ");
}

function bboxToPoints(bbox: [number, number, number, number]): string {
  const [x, y, w, h] = bbox.map(Math.round);
  return `${x},${y} ${x + w},${y} ${x + w},${y + h} ${x},${y + h}`;
}

/** Escape XML special characters */
function esc(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/**
 * Convert COCO JSON (single image) to a well-formed PAGE XML string.
 */
export function cocoToPageXml(cocoJson: COCOJson): string {
  const idToName: Record<number, string> = {};
  for (const c of cocoJson.categories) idToName[c.id] = c.name;

  const image = cocoJson.images[0] ?? {
    id: 0,
    file_name: "unknown",
    width: 0,
    height: 0,
  };
  const anns = cocoJson.annotations.filter((a) => a.image_id === image.id);
  const now = new Date().toISOString();

  const lines: string[] = [];

  lines.push(`<?xml version="1.0" encoding="UTF-8"?>`);
  lines.push(
    `<PcGts xmlns="${PAGE_NS}" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="${SCHEMA_LOC}">`,
  );
  lines.push(`  <Metadata>`);
  lines.push(`    <Creator>Manuscript Layout Analysis</Creator>`);
  lines.push(`    <Created>${now}</Created>`);
  lines.push(`  </Metadata>`);
  lines.push(
    `  <Page imageFilename="${esc(image.file_name)}" imageWidth="${image.width}" imageHeight="${image.height}">`,
  );

  anns.forEach((ann, idx) => {
    const catName = idToName[ann.category_id] ?? "Unknown";
    const [tag, rtype] = REGION_MAP[catName] ?? ["CustomRegion", null];

    let attrs = `id="r_${idx + 1}"`;
    if (rtype) attrs += ` type="${rtype}"`;
    attrs += ` custom="structure {type:${esc(catName)};}"`;

    let points = "";
    if (ann.segmentation?.length > 0 && ann.segmentation[0].length >= 4) {
      points = segToPoints(ann.segmentation[0]);
    } else if (ann.bbox) {
      points = bboxToPoints(ann.bbox);
    }

    lines.push(`    <${tag} ${attrs}>`);
    lines.push(`      <Coords points="${points}"/>`);
    lines.push(`    </${tag}>`);
  });

  lines.push(`  </Page>`);
  lines.push(`</PcGts>`);

  return lines.join("\n");
}

/**
 * Generate PAGE XML from COCO JSON and trigger a download.
 */
export function downloadPageXml(
  cocoJson: COCOJson,
  filename = "page.xml",
): void {
  const xml = cocoToPageXml(cocoJson);
  const blob = new Blob([xml], { type: "application/xml" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
