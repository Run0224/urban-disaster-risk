# 多源城市灾害风险预测

融合雷达、卫星、人口、土地覆盖和地形信息，
使用 GLM-4.6V 进行区域受灾风险预测试验。

## 主要功能

- 卫星与雷达网格对应
- 人口、土地覆盖和地形信息处理
- 灾害记录与区域样本整理
- 雷达与卫星数据缓存
- 区域多源特征提取
- GLM-4.6V 多模态风险推理

## 主要入口

- align_satellite_to_radar.py：卫星网格转换
- align_geography_to_radar.py：人口与土地覆盖聚合
- cache_combined_inputs.py：天气数据缓存
- extract_baseline_features.py：区域特征提取
- extract_approximate_features.py：近似区域范围特征补提取
- run_glm_predictions.py：GLM-4.6V 批量推理

其他脚本包含数据准备、检查和历史试验代码。

## 运行条件

需要 Python，以及 NumPy、Pillow、Matplotlib、
Requests、Rasterio 等依赖，部分脚本还有额外依赖。

GLM 推理服务需要单独部署。
当前默认接口：http://127.0.0.1:8002/v1/chat/completions

## 数据与路径

本仓库只分享代码，不包含原始数据、模型权重和密钥。
代码目前保留原服务器的数据路径，运行前需要调整。
部分脚本依赖前序处理生成的文件，不能直接单独运行。

## 当前状态

这是研究试验代码，尚未整理成一键运行的完整项目。
卫星定位及部分区域范围使用近似信息。
大模型输出概率尚未校准，不用于正式灾害预警。
受灾人口与经济损失预测尚未验证。
