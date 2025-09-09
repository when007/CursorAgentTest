# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

这是一个使用原生 HTML5、CSS3 和 JavaScript 构建的单文件交互式粒子动画系统。应用程序创建视觉吸引人的粒子效果，响应用户输入（鼠标移动和触摸）。

## 开发命令

### 本地开发
```bash
# 启动本地 HTTP 服务器（canvas 正常工作必需）
uv run python -m http.server 8080 --bind 0.0.0.0

# 访问应用程序
# http://localhost:8080/index.html
# http://your-server-ip:8080/index.html
```

### 测试
没有使用正式的测试框架。测试更改：
1. 启动 HTTP 服务器
2. 在浏览器中打开
3. 测试所有粒子效果和交互

## 架构

### 核心组件

**粒子系统** (`index.html:96-186`)
- 基于物理运动的 `Particle` 类
- 具有自动再生的生命周期管理
- 支持多种行为模式

**效果系统** (`index.html:114-148`)
- 四种不同的粒子行为：流动、螺旋、爆炸、波浪
- 通过 UI 按钮切换，立即重置状态
- 每种效果使用不同的力计算和运动模式

**渲染引擎** (`index.html:219-254`)
- 基于 Canvas 的 2D 渲染，具有渐变背景
- 基于接近度的粒子连接
- 发光效果和 Alpha 混合以增强视觉吸引力

**交互系统** (`index.html:256-289`)
- 鼠标和触摸事件处理
- 实时光标位置跟踪
- 响应式 Canvas 调整大小

### 关键技术细节

**物理模拟**
- 基于速度的粒子运动，带阻尼
- 基于鼠标接近度的力场计算
- 边界碰撞检测和响应
- 基于生命周期的粒子生成/销毁

**性能考虑**
- 粒子池最多保持 100 个粒子
- 高效的距离计算用于连接
- RequestAnimationFrame 实现流畅的 60fps 动画
- 最小化 DOM 操作

**效果行为**
- **流动**：向鼠标位置的吸引力
- **螺旋**：围绕鼠标位置的旋转力
- **爆炸**：远离鼠标位置的排斥力
- **波浪**：基于时间和位置的正弦运动

### 代码结构

整个应用程序包含在单个 HTML 文件中，嵌入了 CSS 和 JavaScript。代码按逻辑部分组织：

1. HTML 结构，包含 Canvas 和 UI 元素
2. CSS 样式，具有玻璃形态效果
3. JavaScript 应用程序逻辑
   - Canvas 设置和全局状态
   - Particle 类定义
   - 动画循环
   - 事件处理器
   - 效果切换逻辑

## 开发说明

- 应用程序使用中文作为 UI 文本和注释
- 不需要外部依赖或构建工具
- 响应式设计适用于桌面和移动设备
- 正确处理触摸事件以实现移动交互
- 始终使用 uv 进行 Python 包管理，而不是直接使用 pip
- 使用 uv run python 而不是直接的 python3 命令