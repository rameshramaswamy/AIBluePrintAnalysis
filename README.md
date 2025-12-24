

# AI BluePrint Analysis 

AI BluePrint Analysis is an automated tool designed to parse, analyze, and extract meaningful data from architectural blueprints and technical drawings. By leveraging Computer Vision (CV) and Large Language Models (LLMs), this project transforms static image/PDF plans into actionable digital insights.

##  Features

- **Automated Entity Recognition:** Detects walls, doors, windows, and room labels.
- **OCR Integration:** Extracts text annotations, dimensions, and schedule tables from drawings.
- **Compliance Checking:** (Optional/Planned) Compares blueprint layouts against specific building codes or standards.
- **Vector Conversion:** Assistance in converting raster blueprints into structured data formats (JSON/CSV).
- **Multi-format Support:** Supports PDF, PNG, and JPEG uploads.

##  Tech Stack

- **Language:** Python 3.9+
- **AI/ML:** PyTorch / TensorFlow, OpenAI API (or similar LLM), HuggingFace Transformers.
- **Computer Vision:** OpenCV, LayoutParser, or Detectron2.
- **Interface:** Streamlit / Flask (Update based on your UI choice).

##  Getting Started

### Prerequisites

Ensure you have Python installed and a virtual environment set up.

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/rameshramaswamy/AIBluePrintAnalysis.git
   cd AIBluePrintAnalysis
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Variables:**
   Create a `.env` file in the root directory and add your API keys (if applicable):
   ```env
   OPENAI_API_KEY=your_key_here
   ```

##  Usage

To run the main analysis script:

```bash
python main.py --input path/to/blueprint.pdf --output ./results
```

If using the web interface:
```bash
streamlit run app.py
```

##  Project Structure

```text
├── data/               # Sample blueprints for testing
├── models/             # Pre-trained models or weights
├── src/
│   ├── vision/         # Image processing and CV logic
│   ├── analysis/       # LLM integration and data parsing
│   └── utils/          # Helper functions
├── app.py              # User interface (if applicable)
├── requirements.txt    # Project dependencies
└── README.md
```

##  Contributing

Contributions are welcome! Please follow these steps:
1. Fork the Project.
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`).
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the Branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.


**Ramesh Ramaswamy** - [GitHub Profile](https://github.com/rameshramaswamy)

Project Link: [https://github.com/rameshramaswamy/AIBluePrintAnalysis](https://github.com/rameshramaswamy/AIBluePrintAnalysis)
