# 👃 VLM2SMELL: Olfactory Video Analysis System

> **From Pixels to Molecules:** A Video-to-Scent Translation Pipeline powered by Multimodal LLMs.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)
![Gemini](https://img.shields.io/badge/AI-Google%20Gemini%202.5%20Flash-orange?style=flat-square&logo=google)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Research%20Preview-purple?style=flat-square)

## 📖 Overview

**VLM2SMELL** is a pioneering research framework designed to bridge the gap between **Visual Perception** and **Olfactory Inference**. By leveraging the advanced sequence understanding capabilities of modern Visual Language Models (specifically **Gemini 2.5 Flash**), this system transforms raw video footage into structured chemical and sensory olfactory profiles.

Unlike traditional frame-by-frame analysis, VLM2SMELL treats video as a **continuous temporal sequence**, allowing it to:
1.  **Extract Visual Ground Truth**: Identify objects, states, and thermal cues over time.
2.  **Infer Olfactory Events**: Detect activities that release scent (e.g., *slicing lemon*, *frying steak*, *rain on asphalt*).
3.  **Map to Chemistry**: Translate semantic descriptions into specific odorant molecules (e.g., *Limonene*, *Maillard reaction products*, *Geosmin*).

This project is tailored for **HCI researchers** and developers exploring multi-sensory digital experiences, immersive environments, and digital scent technologies.

---

## 🚀 Key Features

*   **Sequence-First Architecture**: Analyzes full video sequences to capture temporal context (e.g., understanding the transition from "whole onion" $\to$ "chopped onion" $\to$ "sautéed onion").
*   **Visual-Olfactory Separation (VOS)**: Strictly enforces a two-step logic (Visual Evidence $\to$ Chemical Inference) to minimize hallucinations and ground predictions in visual facts.
*   **Chemical Mapping**: Outputs structured data including **Scent Category**, **Descriptors**, **Intensity**, and **Candidate Molecules**.
*   **Ground Truth Preservation**: Automatically extracts and saves indexed frames for manual verification against the generated JSON report.
*   **Gemini 2.5 Flash Integration**: Utilizes the latest efficient multimodal model from Google for fast and accurate long-context understanding.

---

## ⚡ Quick Start

Get up and running in minutes!

1.  **Clone & Install**:
    ```bash
    git clone https://github.com/yourusername/VLM2SMELL.git
    cd VLM2SMELL
    pip install -r requirements.txt
    ```

2.  **Configure API Key**:
    Create a `.env` file and add your Google Gemini API Key:
    ```env
    GOOGLE_API_KEY=your_actual_api_key_here
    ```

3.  **Run Analysis**:
    ```bash
    python3 main.py "test video 1.mp4"
    ```
    *Check the generated `.json` file for the smell report!*

---

## 🛠️ Installation & Setup

### Prerequisites
*   Python 3.9 or higher
*   A Google Cloud Project with the Gemini API enabled
*   An API Key from [Google AI Studio](https://aistudio.google.com/)

### Detailed Steps

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/VLM2SMELL.git
    cd VLM2SMELL
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.  **Environment Configuration:**
    Create a `.env` file in the root directory. You can use the provided example or create a new one.
    ```bash
    echo "GOOGLE_API_KEY=your_api_key_here" > .env
    ```

---

## 💻 Usage

The core script `main.py` handles the entire pipeline: frame extraction, VLM inference, and report generation.

```bash
python3 main.py [video_path] [FPS] [options]
```

### Arguments

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `video_path` | `str` | **Required** | Path to the input video file (e.g., `videos/cooking.mp4`). |
| `FPS` | `int` | `4` | Frames Per Second to extract. Higher FPS = finer detail but higher API cost/latency. |
| `--output` | `str` | `Auto-generated` | Custom path for the output JSON file. |
| `--fps` | `int` | `4` | Alternative flag to specify FPS. |

### Examples

**1. Standard Analysis (Default 4 FPS):**
Good for general activities.
```bash
python3 main.py "input_video.mp4"
```

**2. High-Frequency Analysis (10 FPS):**
Better for fast-paced actions like chopping or rapid chemical reactions.
```bash
python3 main.py "input_video.mp4" 10
```

**3. Custom Output Filename:**
```bash
python3 main.py "input_video.mp4" --output "results/lemon_analysis.json"
```

---

## 📂 Output Structure

The system generates a comprehensive **JSON report** and a folder of **extracted frames**.

### 1. JSON Report
The JSON file contains a structured analysis of the video.

*   **`meta`**: Metadata about the analysis (source video, timestamp, model used).
*   **`visual_timeline`**: High-level event log (e.g., "0.5s: Lemon is sliced").
*   **`frame_log`**: Detailed periodic analysis.

**Example Snippet:**
```json
{
  "timestamp": 2.5,
  "scene": "Kitchen counter close-up",
  "objects": [
    {
      "name": "Lemon",
      "visual_state": "Sliced/Juicy",
      "interaction": "Being squeezed"
    }
  ],
  "scent": {
    "category": "Citrus",
    "descriptors": ["Fresh", "Zesty", "Acidic", "Sharp"],
    "molecules": ["Limonene", "Citral", "beta-Pinene"],
    "intensity": "High",
    "reasoning": "Mechanical action (squeezing) ruptures oil glands in the flavedo (peel), releasing volatile oils."
  }
}
```

### 2. Temporary Frames
A folder named `temp_frames/<video_name>_<timestamp>` is created containing the extracted JPEG images.
> **Note:** These are preserved to serve as the **Ground Truth** for verification. You can manually inspect `frame_000XX.jpg` to verify the VLM's description.

---

## 🏗️ Architecture

The system follows a strict **Two-Step VOS (Visual-Olfactory Separation)** pipeline to ensure accuracy and minimize hallucinations.

```mermaid
graph LR
    A["Input Video"] -->|OpenCV| B["Frame Extraction"]
    B -->|JPEGs| C["Step 1: Visual Analysis"]
    C -->|"Gemini 2.5 Flash"| D["Visual Report (JSON)"]
    D -->|"Step 2: Semantic Translation"| E["Gemini 2.5 Flash"]
    E -->|"Chemical Inference"| F["Final JSON Report"]
```

1.  **Step 1: Visual Understanding via VLM**
    *   **Input**: Sequence of video frames.
    *   **Model**: Gemini 2.5 Flash (Vision + Text).
    *   **Task**: Identify scenes, objects, actions, and physical state changes. *Strictly forbidden from inferring smells.*
    *   **Output**: Pure visual semantic data.
    *   **Reliability Enforcement**: Includes strict coverage validation (>95% duration) and auto-retry logic to prevent hallucinated summaries.

2.  **Step 2: Semantic-to-Chemical Translation via LLM**
    *   **Input**: The structured visual report from Step 1.
    *   **Model**: Gemini 2.5 Flash (Text-only mode).
    *   **Task**: Map visual triggers (e.g., "sliced lemon") to olfactory data (e.g., "Limonene", "Citrus").
    *   **Strategy**: Uses **Guideline-Based Prompting** (Explicit Rules for Intensity, Molecular Complexity, and Causal Reasoning) instead of simple few-shot examples to ensure scientific accuracy across diverse scenarios.
    *   **Output**: The final JSON report with populated `scent` fields.

---

## ❓ Troubleshooting

*   **Error: `Video file not found`**
    *   Ensure the path to your video file is correct. Use absolute paths if unsure.
*   **Error: `Google API key not found`**
    *   Make sure you have created the `.env` file in the root directory and defined `GOOGLE_API_KEY`.
*   **Error: `429 Resource Exhausted`**
    *   You may have hit the rate limit for the Gemini API. Wait a minute and try again, or check your quota in Google AI Studio.
*   **Analysis is too slow?**
    *   Try reducing the FPS (e.g., use `1` or `2` FPS).
    *   Ensure your video is not excessively long (recommended < 2 minutes).

---

## 🤝 Contributing

Contributions are welcome! We are especially looking for:
*   Improved system prompts for better chemical accuracy.
*   Support for other VLM models (e.g., GPT-4o, Claude 3.5).
*   Real-time processing capabilities.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

**Note**: This project uses Google's Generative AI. Ensure you comply with their [usage policies](https://policies.google.com/terms/generative-ai/use-policy).
