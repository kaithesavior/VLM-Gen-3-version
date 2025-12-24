# 👃 VLM2SMELL: Olfactory Video Analysis System

> **From Pixels to Molecules:** A Video-to-Scent Translation Pipeline powered by Multimodal LLMs.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Gemini](https://img.shields.io/badge/AI-Google%20Gemini%201.5-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 📖 Overview

**VLM2SMELL** is a research framework designed to bridge the gap between **Visual Perception** and **Olfactory Inference**. By leveraging the sequence understanding capabilities of modern Visual Language Models (VLMs like Gemini 1.5 Pro/Flash), this system transforms raw video footage into structured chemical and sensory olfactory profiles.

Unlike traditional frame-by-frame analysis, VLM2SMELL treats video as a **continuous temporal sequence**, allowing it to:
1.  **Extract Visual Ground Truth**: Identify objects, states, and thermal cues.
2.  **Infer Olfactory Events**: Detect activities that release scent (e.g., *slicing lemon*, *frying steak*).
3.  **Map to Chemistry**: Translate semantic descriptions into specific odorant molecules (e.g., *Limonene*, *Maillard reaction products*).

This project is designed for **HCI researchers** (e.g., for CHI submissions) exploring multi-sensory digital experiences.

---

## 🚀 Key Features

*   **Sequence-First Architecture**: Analyzes full video sequences to capture temporal context (e.g., "whole onion" $\to$ "chopped onion" $\to$ "sautéed onion").
*   **Visual-Olfactory Separation (VOS)**: Strictly enforces a two-step logic (Visual Evidence $\to$ Chemical Inference) to minimize hallucinations.
*   **Chemical Mapping**: Outputs structured data including **Scent Category**, **Descriptors**, and **Candidate Molecules**.
*   **Ground Truth Preservation**: Automatically extracts and saves indexed frames for verification against the generated JSON report.

---

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/VLM2SMELL.git
    cd VLM2SMELL
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up API Key:**
    Create a `.env` file in the root directory and add your Google Gemini API key:
    ```env
    GOOGLE_API_KEY=your_api_key_here
    ```

---

## 💻 Usage

Run the analysis pipeline on any video file:

```bash
python3 main.py "path/to/video.mp4" [FPS]
```

### Arguments
*   `video_path`: Path to the input video file (Required).
*   `FPS`: (Optional) Frames per second to extract. Default is `4`. Higher FPS = finer granularity but higher token cost.
*   `--output`: (Optional) Custom path for the output JSON file.

### Examples

**Standard Analysis (4 FPS):**
```bash
python3 main.py "test video 1.mp4"
```

**High-Frequency Analysis (10 FPS):**
```bash
python3 main.py "test video 1.mp4" 10
```

**Custom Output Name:**
```bash
python3 main.py "test video 1.mp4" --output "lemon_analysis.json"
```

---

## 📂 Output Structure

The system generates a comprehensive JSON report containing:

1.  **`visual_timeline`**: Key events and state changes (e.g., "0.5s: Lemon is sliced").
2.  **`frame_log`**: Detailed frame-by-frame analysis.

**Example Output Snippet:**
```json
{
  "timestamp": 2.5,
  "scene": "Kitchen counter",
  "objects": [
    {
      "name": "Lemon",
      "visual_state": "Sliced/Juicy",
      "interaction": "Being squeezed"
    }
  ],
  "scent": {
    "category": "Citrus",
    "descriptors": ["Fresh", "Zesty", "Acidic"],
    "molecules": ["Limonene", "Citral", "beta-Pinene"],
    "intensity": "High",
    "reasoning": "Mechanical action (squeezing) ruptures oil glands in peel."
  }
}
```

---

## 🏗️ Architecture

```mermaid
graph LR
    A[Input Video] --> B[Frame Extraction]
    B --> C[Temp Sequence Folder]
    C --> D[VLM Client (Gemini 1.5)]
    D --> E{Analysis Pipeline}
    E --> F[Visual Understanding]
    E --> G[Chemical Translation]
    F & G --> H[JSON Report]
```

1.  **Frame Extraction (`video_processor.py`)**: Converts video into a strictly ordered sequence of images (Ground Truth).
2.  **Sequence Analysis (`vlm_client.py`)**: Sends the entire sequence to the VLM with a specialized system prompt enforcing the VOS protocol.
3.  **Data Validation (`schemas.py`)**: Ensures output adheres to strict Pydantic models.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

**Note**: This project uses Google's Generative AI. Ensure you comply with their usage policies and rate limits.
