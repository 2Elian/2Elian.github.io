---
title: ElianFactory: Fine-tuning large language models on Windows made easy
summary: This project developed a tool that can perform fine-tuning training of large models on Windows systems. The fine-tuning inference framework is developed based on the transformers library.
tags:
  - Application
date: 2022-01-01
---

本项目开发了一款可以在Windows系统上进行大模型微调训练的工具，微调推理框架基于transformers库进行开发。

如果您是一名LLM工程师，您可以自行构建您的微调推理代码。
如果您熟悉Docker与Linux虚拟机的安装与操作您可以选择[LlamaFactory](https://github.com/hiyouga/LLaMA-Factory)进行LLM微调与推理。但如果您是一名小白，且仅有Windows操作系统，[ElianFactory](https://github.com/2elian/Elian-Factory)是您的最佳选择。

[ElianFactory项目地址](https://github.com/2elian/Elian-Factory)

## ElianFactory的优势

- 🖥️ **基于Windows系统开发**：ElianFactory完全支持原生态Windows系统，无需部署Linux虚拟机即可进行大语言模型的微调和推理。
- 🚀 **直观的操作页面**：ElianFactory是一个无需构建任何代码的工具，您只需进行简单的配置即可进行LLM的微调与推理。
- 🔍 **训练支持**：在ElianFactory-V1.0.0中，我们只支持SFT训练。在后续的版本中，我们会将DPO、PPO、GRPO等功能集成到ElianFactory中，并会考虑更高效的微调推理框架。

## 演示视频

![演示地址](./video.gif)