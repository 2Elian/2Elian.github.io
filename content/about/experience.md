---
# An instance of the Experience widget.
# Documentation: https://docs.hugoblox.com/page-builder/
widget: experience

# This file represents a page section.
headless: true

# Order that this section appears on the page.
weight: 20

title: Experience
subtitle:

# Date format for experience
#   Refer to https://docs.hugoblox.com/customization/#date-format
date_format: Jan 2006

# Experiences.
#   Add/remove as many `experience` items below as you like.
#   Required fields are `title`, `company`, and `date_start`.
#   Leave `date_end` empty if it's your current employer.
#   Begin multi-line descriptions with YAML's `|2-` multi-line prefix.
experience:
  - title: AIGC算法实习生
    company: Vivo
    company_url: ''
    company_logo: org-gc
    location: 中国 - 南京
    date_start: '2026-06-01'
    date_end: ''
    description: |2-
        工作内容: 深度参与Vivo自研的内容智能创作平台，负责多模态大模型/语言模型的业务化对齐微调训练、Agent工作流设计与搭建、项目工程落地开发(基于Python、Java实现完整的后端开发系统)
        
        * DeepSeek/Qwen模型的SFT与RL训练
        * 针对训练好的模型，负责对其量化与部署，提升推理的性能与效率
        * 基于ComfiUI设计工作流，并基于FastAPI、SpringBoot等框架构建完整的后端业务化系统

  - title: 大模型开发工程师(实习)
    company: 南信大苏州影像技术有限公司
    company_url: ''
    company_logo: org-x
    location: 中国 - 南京
    date_start: '2024-07-10'
    date_end: '2024-12-12'
    description: |2-
        实习期间我的工作内容为：主导设计面向智慧气象与智能金融领域的大模型应用系统，其中包括
        * LLM的微调训练
        * LLM的推理加速
        * RAG架构设计与性能优化

design:
  columns: '1'
---
