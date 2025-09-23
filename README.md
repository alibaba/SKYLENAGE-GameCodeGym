# V-GameGym

A comprehensive evaluation framework for automatically generating and assessing Pygame-based games using Large Language Models (LLMs).

## 🎮 Overview

V-GameGym is an open-source project designed to evaluate and benchmark the capabilities of Large Language Models in generating functional, playable games using the Pygame library. The framework provides tools for automatic game generation, execution, evaluation, and visual recording of gameplay.

## ✨ Features

- **Automatic Game Generation**: Generate Pygame games from natural language requirements using LLMs
- **Game Evaluation**: Comprehensive evaluation system with scoring metrics
- **Screenshot Recording**: Automated screenshot capture during game execution
- **Video Generation**: Create gameplay videos from recorded screenshots
- **Testset Management**: Built-in testset with diverse game examples
- **Multi-processing Support**: Parallel processing for efficient evaluation

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

## 🚀 Getting Started

### Prerequisites

- Python 3.1+
- Pygame
- OpenAI API access or compatible LLM endpoint

### Required Dependencies

```bash
pip install pygame numpy pillow openai tqdm jsonlines
```

### Configuration

1. Edit `config/config.json` to set your LLM API configuration:

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

## 📊 Usage

### Game Generation

Generate Pygame games from text requirements:

```bash
python generate_pygame_codes.py --config config/config.json --input requirements.jsonl --output generated_games.jsonl
```

### Game Evaluation

Evaluate generated games with automatic execution and recording:

```bash
python game_evaluator.py --input games.jsonl --output results.jsonl --record-screenshots --generate-videos
```

### Screenshot Recording

Record gameplay screenshots and generate videos:

```bash
python screenshot_recorder.py --game-file game.py --duration 10 --fps 5
```

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

## 🔧 Key Components

### Code Generator (`generate_pygame_codes.py`)
- Interfaces with LLMs to generate game code
- Handles prompt engineering for game requirements
- Supports batch processing with multiprocessing
- Includes error handling and retry mechanisms

### Screenshot Recorder (`screenshot_recorder.py`)
- Automated screenshot capture during game execution
- Video generation from screenshot sequences
- Configurable frame rates and recording duration
- Optimized file formats for storage efficiency

### Game Evaluator (`game_evaluator.py`)
- Executes generated games in isolated environments
- Captures runtime errors and execution metrics
- Records screenshots at regular intervals
- Generates quality scores based on execution success

## 🤝 Contributing

We welcome contributions to improve V-GameGym! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is released under the MIT License. See the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use V-GameGym in your research, please cite:

```bibtex
```

## 🙏 Acknowledgments

- Thanks to the Pygame community for the excellent game development framework
- OpenAI and other LLM providers for enabling automated code generation
- Contributors and researchers in the field of automated programming

---

**Note**: This is an open-source research project. Game generation quality may vary depending on the LLM model and configuration used.