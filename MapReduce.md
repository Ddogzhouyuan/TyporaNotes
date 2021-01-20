# MapReduce

### MapReduce基本定义

MapReduce是面向大数据并行处理的计算模型、框架和平台。

* MapReduce是一个基于集群的高性能并行计算平台（Cluster Infrastructure）
* MapReduce是一个并行计算运行的软件框架（Software Framework）
* MapReduce是一个并行程序设计模型与方法（Programming Model）

### MapReduce模型简介

* MapReduce将复杂的、运行于大规模集群上的并行计算过程高度抽象成了两个函数：Map和Reduce，现实世界中大部分任务都可以抽象成map和reduce
* MapReduce采用分而治之的策略，存储在HDFS的大规模数据集，按照数据块切分成多个map任务并行处理
* 易于编程、高扩展性、高容错性

### Map和Reduce函数

|        | 输入           | 输出         |
| ------ | -------------- | ------------ |
| Map    | (k1, v1)       | List(k2, v2) |
| Reduce | (k2, List(v2)) | (k3, v3)     |

* 将数据集解析成一批kv对，用map进行处理

* 每一个kv对会输出一批新的kv作为中间结果

例如，word count的一个过程，map输入（行号，"a a a"），输出List(("a", 1), ("a", 1), ("a", 1))；reduce输入（"a", (1, 1, 1)）输出（"a", 3）。

![](assets\MapReduce数据流.png)

## MapReduce作业运行机制

整个MapReduce作业的运行过程如下图所示：

![](assets\MapReduce作业工作原理.png)

图中包含5个主要的部分：

* 客户端，用于提交MapReduce作业
* YARN的资源管理器（ResourceManager），负责集群资源的分配
* YARN的节点管理器（NodeManager），负责启动和监控集群中机器上的容器（container）
* Application Master，负责协调运行MapReduce作业的任务，它和MapReduce任务都在容器中运行，这些容器由资源管理器分配并由节点管理器管理。
* 分布式文件系统（HDFS），用于各部分之间作业文件共享。

### 作业提交

Job的submit()方法创建一个内部的JobSubmitter实例，调用其submitJobInternal()方法（图中步骤1）。提交作业后，waitForCompletion()方法内部每秒查询作业进度，更新进度到控制台。

JobSubmitter实现作业提交过程：

1. 向资源管理器申请一个新的Application ID，作为MapReduce作业ID（步骤2）
2. 检查作业的输出（作业校验）。例如没有指定输出目录或输出目录已经存在，不提交作业，MapReduce抛出错误
3. 计算作业的输入分片（作业校验）。如果不能计算分片（比如输入路径不存在），不提交作业，MapReduce抛出错误
4. 拷贝执行作业需要的资源到共享文件系统的以作业ID命名的目录中（步骤3），包括Jar包、配置文件、计算好的输入切片。Jar包的副本数由mapreduce.client.submit.file.replication控制，默认是10，在运行任务的时候，节点管理器可以访问这些副本。
5. 通过资源管理器的submitApplication()方法提交作业（步骤4）

### 作业初始化

* YARN调度器为作业请求分配一个容器（步骤5a），资源管理器通过容器所在节点上的节点管理器在该容器中启动Application Master进程（步骤5b）
* MapReduce作业的Application Master是一个Java应用，主类是MRAppMaster，作用有接受任务的进度和完成报告（步骤6）；它还接收共享文件系统的、客户端计算的输入分片（步骤7）。
* 对每个分片创建map任务以及确定的多个reduce任务对象。
* 分配任务ID

### 任务分配

* Application Master为该作业中所有的map和reduce任务向资源管理器请求容器（步骤8）
* 为map任务发起的请求会首先进行，并且请求的优先级高于reduce任务，直到有5%的map任务完成，为reduce任务的请求才会发出
* reduce任务可以在集群任意位置运行，而map任务有数据本地化的限制。理想情况下，任务是数据本地化的（data local），即任务和分片在同一节点上运行。其他模式，例如机架本地化（rack local），效率会低于数据本地化。

### 任务执行

* 一旦资源管理器在一个节点上的一个容器中为一个任务分配了资源，Application Master与节点管理器通信来启动容器（步骤9）
* 任务通过一个Java应用程序执行，该程序主类是YarnChild。
* 在运行之前，需要从文件共享系统本地化任务需要的资源（jar包、配置文件等）
* 运行map或reduce任务

### 进度状态更新



### 作业完成

## 作业失败

### 任务失败

### Application Master运行失败

### 节点管理器运行失败

### 资源管理器运行失败



