# AlexNet

## AlexNet网络结构

<img src="http://img.blog.itpub.net/blog/2018/07/25/1fbe544951a6129b.jpeg?x-oss-process=style/bb">

AlexNet 不算池化层总共有 8 层，前 5 层为卷积层，其中第一、第二和第五层卷积都包含了一个最大池化层，后三层为全连接层。所以 AlexNet 的简略结构如下： 
输入>卷积>池化>卷积>池化>卷积>卷积>卷积>池化>全连接>全连接>全连接>输出

各层的结构和参数如下：  
C1层是个卷积层，其输入输出结构如下：  
输入： 227 x 227 x 3  滤波器大小： 11 x 11 x 3   滤波器个数：96  
输出： 55 x 55 x 96  

P1层是C1后面的池化层，其输入输出结构如下：  
输入： 55 x 55 x 96  滤波器大小： 3 x 3   滤波器个数：96  
输出： 27 x 27 x 96  

C2层是个卷积层，其输入输出结构如下：   
输入： 27 x 27 x 96  滤波器大小： 5 x 5 x 96   滤波器个数：256   
输出： 27 x 27 x 256  

P2层是C2后面的池化层，其输入输出结构如下：   
输入： 27 x 27 x 256  滤波器大小： 3 x 3   滤波器个数：256   
输出： 13 x 13 x 256  

C3层是个卷积层，其输入输出结构如下：   
输入： 13 x 13 x 256  滤波器大小： 3 x 3 x 256   滤波器个数：384   
输出： 13 x 13 x 384  

C4层是个卷积层，其输入输出结构如下：   
输入： 13 x 13 x 384  滤波器大小： 3 x 3 x 384   滤波器个数：384   
输出： 13 x 13 x 384  

C5层是个卷积层，其输入输出结构如下：   
输入： 13 x 13 x 384  滤波器大小： 3 x 3 x 384    滤波器个数：256   
输出： 13 x 13 x 256  

P5层是C5后面的池化层，其输入输出结构如下：   
输入： 13 x 13 x 256  滤波器大小： 3 x 3     滤波器个数：256   
输出： 6 x 6 x 256  

F6层是个全连接层，其输入输出结构如下：   
输入：6 x 6 x 256   
输出：4096

F7层是个全连接层，其输入输出结构如下：   
输入：4096   
输出：4096  

F8层也是个全连接层，即输出层，其输入输出结构如下： 
输入：4096 
输出：1000

在论文中，输入图像大小为 224 x 224 x 3，实际为 227 x 227 x 3。各层输出采用 relu 进行激活。前五层卷积虽然计算量极大，但参数量并不如后三层的全连接层多，但前五层卷积层的作用却要比全连接层重要许多。   