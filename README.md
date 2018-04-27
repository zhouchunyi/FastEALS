# sigir16-eals
Experiments codes for SIGIR'16 paper "Fast Matrix Factorization for Online Recommendation with Implicit Feedback "

### 项目说明
该项目是基于何向南的fastEALS算法的应用和修改。

1、algorithms/MF_fastALS_WRMF是对MF_fastALS的扩充，添加了参数设置选择；

2、main/main_XiamiOnline是针对虾米数据的实验环境：

    （1）train文件：单个文件，和librec的训练文件格式一致，userId<space>itemId<space>freq
    （2）test文件夹：对每个用户的播放记录文件，文件名为<userId>.txt，每行内容为timestamp<\t>itemId