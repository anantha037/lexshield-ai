import os
import json
from pathlib import Path
from datasets import load_dataset

# Add your token here
HF_TOKEN = os.getenv("HF_TOKEN")

OUTPUT_DIR = Path("data/raw/judgments")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def download_iltur_judgments():
    """
    IL-TUR dataset - using correct config names.
    'summ' = judgment summarization data (full SC cases + summaries)
    'bail' = bail prediction cases (criminal law focused)
    """
    all_judgments = []

    # Config 1: summ — full SC judgment texts with summaries
    print("Downloading IL-TUR 'summ' (judgment summaries)...")
    try:
        ds = load_dataset("Exploration-Lab/IL-TUR", "summ", token=HF_TOKEN)
        split = "train" if "train" in ds else list(ds.keys())[0]
        for item in ds[split]:
            doc = item.get("document", "") or item.get("text", "")
            text = " ".join(doc) if isinstance(doc, list) else doc
            if text.strip():
                all_judgments.append({
                    "doc_id": str(item.get("id", len(all_judgments))),
                    "text": text,
                    "summary": item.get("summary", ""),
                    "court": "Supreme Court of India",
                    "doc_type": "judgment",
                    "source_config": "summ"
                })
        print(f"    Got {len(all_judgments)} judgment docs from 'summ'")
    except Exception as e:
        print(f"    summ failed: {e}")

    # Config 2: bail — criminal bail cases
    print("Downloading IL-TUR 'bail' (bail cases)...")
    bail_count = 0
    try:
        ds = load_dataset("Exploration-Lab/IL-TUR", "bail", token=HF_TOKEN)
        split = "train" if "train" in ds else list(ds.keys())[0]
        for item in ds[split]:
            doc = item.get("facts", "") or item.get("text", "") or item.get("document", "")
            text = " ".join(doc) if isinstance(doc, list) else str(doc)
            if text.strip():
                all_judgments.append({
                    "doc_id": f"bail_{len(all_judgments)}",
                    "text": text,
                    "court": "Supreme Court of India",
                    "doc_type": "judgment",
                    "source_config": "bail"
                })
                bail_count += 1
        print(f"    Got {bail_count} bail cases")
    except Exception as e:
        print(f"    bail failed: {e}")

    if all_judgments:
        output_path = OUTPUT_DIR / "iltur_judgments.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_judgments, f, ensure_ascii=False, indent=2)
        print(f"\nSaved {len(all_judgments)} total IL-TUR docs → {output_path}")
    else:
        print("No IL-TUR data saved — sc_prechunked.json is sufficient, continuing.")

def download_pre_chunked_judgments():
    """
    Pre-chunked SC Judgments — already processed, ready to use.
    """
    print("\nDownloading pre-chunked SC Judgments dataset...")
    
    try:
        # Added token, removed trust_remote_code
        dataset = load_dataset("vihaannnn/Indian-Supreme-Court-Judgements-Chunked", token=HF_TOKEN)

        chunks = []
        split = list(dataset.keys())[0]
        for item in dataset[split]:
            text = item.get("text") or item.get("chunk") or ""
            if text.strip():
                chunks.append({
                    "text": text,
                    "source": item.get("source", "Supreme Court of India"),
                    "doc_type": "judgment",
                    "chunk_id": f"hf_sc_{len(chunks):05d}"
                })

        output_path = OUTPUT_DIR / "sc_prechunked.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        print(f"Saved {len(chunks)} pre-chunked judgment chunks to {output_path}")
        
    except Exception as e:
        print(f"Failed to download pre-chunked dataset: {e}")

if __name__ == "__main__":
    download_iltur_judgments()
    # download_pre_chunked_judgments()