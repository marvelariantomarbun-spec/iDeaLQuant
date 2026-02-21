"""Extract text and images from the crash evidence docx."""
import os
from docx import Document

doc_path = r"D:\Projects\IdealQuant\reference\Hata Görüntüsü.docx"
out_dir = r"C:\Users\ttevf\.gemini\antigravity\brain\0d503404-8318-494b-9c49-d225bcb6d78a"

doc = Document(doc_path)

# Extract text
print("=== DOCUMENT TEXT ===")
for i, p in enumerate(doc.paragraphs):
    if p.text.strip():
        print(f"[P{i}] {p.text}")

# Extract images
img_count = 0
for rel in doc.part.rels.values():
    if "image" in rel.reltype:
        img_count += 1
        img_data = rel.target_part.blob
        ext = os.path.splitext(rel.target_part.partname)[1]
        img_name = f"crash_evidence_{img_count}{ext}"
        img_path = os.path.join(out_dir, img_name)
        with open(img_path, 'wb') as f:
            f.write(img_data)
        print(f"[IMG] Saved: {img_name} ({len(img_data)} bytes)")

print(f"\nTotal images: {img_count}")
