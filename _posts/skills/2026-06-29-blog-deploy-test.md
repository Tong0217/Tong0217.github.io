---
layout: post
title: 博客自动部署测试
category: 技能
tags: [github-actions, ecs]
description: 验证博客从 GitHub Actions 自动构建并部署到 ECS 的流程
---

这篇文章用于验证博客的自动部署链路。

当前流程是：提交代码到 `main` 分支后，GitHub Actions 自动构建 Jekyll 静态站点，并通过受限的 `blog-deploy` 用户同步到 ECS 上的 `/var/www/blog` 目录。

如果你能在 `https://tongch.top/blog` 看到这篇文章，就说明自动部署流程已经生效。

Secret 修正后，重新触发一次自动部署验证。
