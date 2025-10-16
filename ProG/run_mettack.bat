@echo off
REM 设置环境变量以解决OpenMP冲突问题
SET KMP_DUPLICATE_LIB_OK=TRUE

REM 激活conda环境（如果需要）
call conda activate GPL

REM 运行mettack脚本
python DeepRobust/examples/graph/test_mettack.py --dataset cora --ptb_rate 0.05

REM 暂停以查看结果
pause