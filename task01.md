# 学习中遇到的最大的难点

1. 如何开始任务

尽管已经给予了两个baseline，但在实际运行中，仅仅是跑通baseline就是面临的头等大事。

对于水神的baseline，尽管水神已经打包了整个case，但似乎需要用linux系统运行，而自己对于搭建运行linux系统十分生疏（完全不懂）。

首先尝试了用docker，发现需要首先学习docker，后又尝试使用pycharm，但也遇到了一堆问题。经过百度，发现可以安装windows下的子系统Ubuntu

安装完成后，窃以为可以安稳的运行系统，却又发现下载后的文件夹不知道在哪里，在文件管理器中找不到文件>>>现在的解决方案是，自己手动先把文件git clone下来，放在desktop，然后通过Ubuntu系统进入到c盘下desktop中的文件，再去尝试一下。



总结来说，确实是一次不容易的学习之路。从开始就充满了荆棘。



尝试用百度的飞桨平台，采用CPU模式发现内存不够，开启GPU模式下时，总是人机验证失败，无法进入。

在nootbook中，若要加载如wget、git clone等命令时，需要在代码前加！
