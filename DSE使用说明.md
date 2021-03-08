# 使用教程

本工具用于探索不同数据流和硬件参数配置（计算单元和存储单元的数量）对卷积神经网络的计算性能的影响。

## 环境需求

需要的环境非常简单，需要在Linux系统环境下运行，建议使用Anaconda进行安装。

- python >= 3.5
- os
- subprocess
- math
- comfigparser
- tdqm
- absl-py

## 参数设置

### 1. 硬件架构参数设置

1. 硬件架构参数文件位于`./configs`之下, 文件格式为`.cfg`

2. 硬件参数设置如下

<img src="https://i.loli.net/2021/03/04/KUiMymc1Nf3sknT.png" alt="image-20210304224140648" style="zoom: 67%;" />

| 说明                                                 | 参数名                   |
| ---------------------------------------------------- | ------------------------ |
| 运行的参数配置名, 不同配置不可重复                   | run_name                 |
| 脉动阵列的高(MAC单元数量)                            | ArrayHeight              |
| 脉动阵列的宽(MAC单元数量)                            | ArrayWidth               |
| 输入特征图缓存大小(单位KB)                           | IfmapSRAMSz              |
| 权重缓存大小(单位KB)                                 | FilterSRAMSz             |
| 输出特征图缓存大小(单位KB)                           | OfmapSRAMSz              |
| 输入/权重/输出存储的基地址位置, 只需要相差足够多即可 | Ifmap/Filter/OfmapOffset |
| 包含is(输入固定)/ws(权重固定)/os(输出固定)三中数据流 | Dataflow                 |

3. 指定参数文件

   将`scale.py`中`flags.DEFINE_string("arch_config","./configs/scale.cfg","file where we are getting our architechture from")`中对应的硬件参数文件地址修改为之前设置的参数文件即可。

### 2. CNN网络参数设置

1. 网络参数文件位于`./topologies/conv_nets`, 文件格式为`.csv`, 用于描述CNN网络的拓扑结构。

2. 网络参数的设置如下

   ![image-20210304230202031](https://i.loli.net/2021/03/04/GEwbU1n2APhWmdy.png)

   需要描述网络每一层的名称，网络拓扑参数，需要注意的是，本工具只能支持卷积层的探索（全连接层可以视为特殊的卷积层，只需要将对应参数改为1）, 不同的参数之间以`,`进行分割(上图中为表格视图, 下图为文本视图)

   ![image-20210304232345179](https://i.loli.net/2021/03/04/wu4UgesHyMxaqOY.png)

3. 指定参数文件

   将`flags.DEFINE_string("network","./topologies/conv_nets/mobilenet.csv","topology that we are reading")`中对应的硬件参数文件地址修改为之前设置的参数文件即可。

## 运行

### 1. 单次运行

- 将`scale.py`中`main()`函数一项中的`sweep 设置为 False`
- 在命令行中运行`python scale.py `即可运行, 此时运行的结果是按照固定的硬件参数对网络进行搜索, 得到运行的结果。

### 2. 搜索

- 将`scale.py`中`main()`函数一项中的`sweep 设置为 True`

- 修改硬件参数文件中的硬件参数如下

  <img src="https://i.loli.net/2021/03/04/DrhzoWQtsSXUgim.png" alt="image-20210304231454293" style="zoom:67%;" />

- 以图中的`IfmapSramSz`为例，以`,`作为分界，左侧的数字`2`为扫描的起点`32`为扫描的终点, 同时, 搜索模式还会对数据流进行搜索, 可以修改`scale.py`中的`run_sweep()`函数来限定需要搜索的数据流

- 在命令行中运行`python scale.py `即可运行, 此时运行的结果将会对这些参数进行扫描

- 需要注意的是，如果在运行时进行了搜索将会耗费大量的时间

## 输出

输出文件会存放在生成的`./output`路径下, 目录名与硬件参数文件中指定的`run_name`相同，输出包括

- 逐层的运行时间和平均利用率
- 逐层最大DRAM带宽需求
- 逐层平均DRAM带宽需求
- 逐层数据访问量(输入/权重/输出)

