[![Home](https://img.shields.io/badge/Homepage-🏠-blue.svg)](https://v-gamegym.github.io/index.html)
[![Leaderboard](https://img.shields.io/badge/Leaderboard-🏆-brightgreen.svg)](https://v-gamegym.github.io/leaderboard.html)
[![Paper](https://img.shields.io/badge/Paper-📄-b31b1b.svg)](https://arxiv.org/abs/2509.20136) 
[![Code](https://img.shields.io/badge/Code-💻-black.svg)](https://github.com/alibaba/SKYLENAGE-GameCodeGym/) 

[![Dataset-HF](https://img.shields.io/badge/Data-Huggingface-orange.svg)](https://huggingface.co/datasets/alibabagroup/SKYLENAGE-GameCodeGym) 
[![Dataset-ModelScope](https://img.shields.io/badge/Data-ModelScope-green.svg)](https://modelscope.cn/datasets/Alibaba-DT/SKYLENAGE-GameCodeGym) 
[![Platform](https://img.shields.io/badge/Platform-SKYLENAGE-blue.svg)](https://skylenage.alibaba-inc.com/sla/home) 

# 🎮 SKYLENAGE-GameCodeGym

**SKYLENAGE-GameCodeGym (V-GameGym)** is an open-source benchmark designed to evaluate and measure the capabilities of Large Language Models (LLMs) in generating **functional, playable, and visually rich games** with the Pygame library.  
The framework provides a complete pipeline for **automatic game generation, execution, evaluation, and gameplay recording**, bridging the gap between code generation accuracy and real-world game development workflows.  

---

## ✨ Features

- **Automatic Game Generation**: Convert natural language requirements into runnable Pygame code with LLMs.  
- **Comprehensive Game Evaluation**: Built-in scoring metrics for functionality, playability, and execution.  
- **Visual Recording**: Automated **screenshots and gameplay videos** during execution.  
- **Testset Management**: Includes a curated dataset with **2,219 game samples across 100 clusters**.  
- **Parallel Processing**: Multiprocessing support for efficient large-scale evaluation.  

---

## 📁 Project Structure

```
V-GameGym-opensource/
├── game_evaluator.py          # Main evaluation script
├── generate_pygame_codes.py   # Game generation utilities
├── screenshot_recorder.py     # Screenshot and video recording
├── config/
│   └── config.json           # LLM client configuration
├── gamegym_testset/
│   ├── gamegym_testset.jsonl # Test cases dataset
│   └── files/                # Generated game files and media
└── V_GameGym.pdf             # Research paper
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+  
- Pygame  
- OpenAI API access or compatible LLM endpoint  

### Installation
```bash
pip install pygame numpy pillow openai tqdm jsonlines
```

### Configuration
Edit `config/config.json` to configure your LLM API:
```json
{
  "client_config": {
    "api_key": "your-api-key",
    "base_url": "your-llm-endpoint",
    "timeout": 7200,
    "max_retries": 10
  },
  "chat_config": {
    "model": "your-model-name",
    "temperature": 0.7,
    "max_tokens": 8192
  }
}
```

---

## 📊 Usage

### 1. Game Generation
```bash
python generate_pygame_codes.py --config config/config.json --input requirements.jsonl --output generated_games.jsonl
```

### 2. Game Evaluation
```bash
python game_evaluator.py --input games.jsonl --output results.jsonl --record-screenshots --generate-videos
```

### 3. Screenshot & Video Recording
```bash
python screenshot_recorder.py --game-file game.py --duration 10 --fps 5
```

---

## 🎯 Testset

The project includes a comprehensive testset (`gamegym_testset/gamegym_testset.jsonl`) with diverse game examples:

- **Puzzle Games**: Sliding puzzle, Tetris-style games
- **Action Games**: Frogger-like crossing games, dodge games
- **Sports Games**: Pong-style paddle games
- **Arcade Games**: Various classic arcade game implementations

Each test case includes:
- Game requirements description
- Generated Python code
- Execution results and metadata
- Screenshots and gameplay videos

---

## 🔧 Key Components

### Code Generator (`generate_pygame_codes.py`)
- Interfaces with LLMs for code generation  
- Includes batch processing, error handling, and retries  

### Screenshot Recorder (`screenshot_recorder.py`)
- Captures screenshots during execution  
- Converts image sequences into gameplay videos  

### Game Evaluator (`game_evaluator.py`)
- Runs games in isolated environments  
- Records errors, screenshots, and evaluation metrics  

---

## 🤝 Contributing

We welcome contributions! Please:  
1. Fork the repository  
2. Create a feature branch (`git checkout -b feature/amazing-feature`)  
3. Commit changes (`git commit -m 'Add amazing feature'`)  
4. Push the branch (`git push origin feature/amazing-feature`)  
5. Open a Pull Request  

---

## 📄 License

This project is released under the **Apache License 2.0 License**. See the [LICENSE](LICENSE) file for details.  

---

## 📚 Citation

If you use **V-GameGym** in your research, please cite:  

```bibtex
@misc{zhang2025vgamegymvisualgamegeneration,
  title     = {V-GameGym: Visual Game Generation for Code Large Language Models}, 
  author    = {Wei Zhang and Jack Yang and Renshuai Tao and Lingzheng Chai and Shawn Guo and Jiajun Wu and Xiaoming Chen and Ganqu Cui and Ning Ding and Xander Xu and Hu Wei and Bowen Zhou},
  year      = {2025},
  eprint    = {2509.20136},
  archivePrefix = {arXiv},
  primaryClass  = {cs.SE},
  url       = {https://arxiv.org/abs/2509.20136}
}
```

---

## 🙏 Acknowledgments

- Thanks to the **Pygame community** for the excellent framework  
- **OpenAI and other LLM providers** for enabling automated code generation  
- All contributors and researchers advancing automated programming  

---

🔗 **Official Website**: [Skylenage Benchmark Platform](https://skylenage.alibaba-inc.com/sla/home)  
📧 **Contact Us**: skylenage@service.alibaba.com  
